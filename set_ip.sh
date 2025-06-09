#!/bin/bash
# Enhanced Network Interface Management Script
# Usage: sudo ./net_manager.sh --action [set|unset] --ip [1|2]

# Interface list with 8 ports
interfaces=("ib7s400p0" "ib7s400p1" "ib7s400p2" "ib7s400p3"
            "ib7s400p4" "ib7s400p5" "ib7s400p6" "ib7s400p7")

# Initialize parameters
action=""
ip_mode=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --action)
            action="$2"
            shift 2
            ;;
        --ip)
            ip_mode="$2"
            shift 2
            ;;
        *)
            echo -e "\033[31mError: Unknown parameter $1\033[0m" >&2
            exit 1
            ;;
    esac
done

# Validate root privileges
if [ "$(id -u)" != "0" ]; then
    echo -e "\033[31mError: Root privileges required\033[0m" >&2
    exit 1
fi

# IP Configuration Function
configure_ips() {
    for index in "${!interfaces[@]}"; do
        local interface=${interfaces[$index]}
        local ip_address="100.100.${index}.${ip_mode}/24"
        
        echo -e "\n\033[33mConfiguring ${interface} => ${ip_address}\033[0m"
        
        # Clear existing configuration
        ip addr flush dev $interface 2>/dev/null
        sleep 1 
        # Apply new IP settings
        if ip addr add $ip_address dev $interface; then
            echo -e "\033[32m[Success] IP assigned\033[0m"
            
            # Activate interface
            if ip link set $interface up; then
                echo -e "\033[32m[Status] Interface activated\033[0m"
            else
                echo -e "\033[31m[Error] Activation failed\033[0m" >&2
            fi
        else
            echo -e "\033[31m[Error] IP assignment failed\033[0m" >&2
        fi
    done
}

# IP Removal Function
unset_ips() {
    for interface in "${interfaces[@]}"; do
        echo -e "\n\033[33mClearing configuration on ${interface}\033[0m"
        
        # Remove all IP addresses
        ip addr flush dev $interface 2>/dev/null
        sleep 1        
        # Deactivate interface
        if ip link set $interface down; then
            echo -e "\033[32m[Success] Interface deactivated\033[0m"
        else
            echo -e "\033[31m[Error] Deactivation failed\033[0m" >&2
        fi
    done
}

# Parameter validation logic
if [[ -z "$action" ]]; then
    echo -e "\033[31mError: Must specify --action parameter\033[0m" >&2
    exit 1
fi

# Execute requested action
case $action in
    "set")
        if ! [[ "$ip_mode" =~ ^[1-2]$ ]]; then
            echo -e "\033[31mError: --ip must be 1 or 2\033[0m" >&2
            exit 1
        fi
        configure_ips
        ;;
    "unset")
        unset_ips
        ;;
    *)
        echo -e "\033[31mError: Invalid action specified\033[0m" >&2
        exit 1
        ;;
esac

# Status display function
show_status() {
    echo -e "\n\033[36mCurrent Network Status:\033[0m"
    printf "%-12s %-18s %-8s %s\n" "Interface" "IP Address" "State" "Physical Link"
    
    for interface in "${interfaces[@]}"; do
        # Retrieve interface information
        local ip_addr=$(ip -o addr show $interface 2>/dev/null | awk '/inet / {print $4}')
        local link_state=$(ip -o link show $interface 2>/dev/null | awk '{print $9}' | tr -d :)
        local phy_state=$(ethtool $interface 2>/dev/null | awk -F': ' '/Link detected/ {print $2}')
        
        # Set status colors
        local state_color="\033[31m"  # Red for down state
        [ "$link_state" = "UP" ] && state_color="\033[32m"
        
        # Format output
        printf "%-12s %-18s ${state_color}%-8s\033[0m %s\n" \
            "$interface" \
            "${ip_addr:-No IP}" \
            "[${link_state^^}]" \
            "${phy_state:-Unknown}"
    done
}

# Display final status
show_status
