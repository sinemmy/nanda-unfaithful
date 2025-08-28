#!/usr/bin/env python3
"""
Auto-terminate vast.ai instance after experiments complete.
Add this to your experiment script to avoid overcharging.

The vastai package should be installed automatically via requirements.txt
If not: pip install vastai
Then configure: vastai set api-key YOUR_API_KEY
"""

import subprocess
import json
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path

def get_instance_id_from_ip(ip_address):
    """Get vast.ai instance ID from IP address."""
    try:
        result = subprocess.run(
            ["vastai", "show", "instances", "--raw"],
            capture_output=True,
            text=True,
            check=True
        )
        
        instances = json.loads(result.stdout)
        for instance in instances:
            if instance.get("public_ipaddr") == ip_address:
                return instance.get("id")
        
        print(f"Warning: Could not find instance with IP {ip_address}")
        return None
    except Exception as e:
        print(f"Error getting instance ID: {e}")
        return None

def destroy_instance(instance_id):
    """Destroy a vast.ai instance."""
    try:
        result = subprocess.run(
            ["vastai", "destroy", "instance", str(instance_id)],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ Instance {instance_id} destroyed successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to destroy instance: {e}")
        return False

def monitor_and_terminate(ip_address, port, max_runtime_hours=4, check_interval_minutes=5):
    """
    Monitor experiments and auto-terminate when complete or after max runtime.
    
    Args:
        ip_address: vast.ai instance IP
        port: SSH port
        max_runtime_hours: Maximum hours before force termination
        check_interval_minutes: How often to check status
    """
    
    start_time = datetime.now()
    max_runtime = timedelta(hours=max_runtime_hours)
    check_interval = timedelta(minutes=check_interval_minutes)
    
    # Get instance ID
    instance_id = get_instance_id_from_ip(ip_address)
    if not instance_id:
        print("Could not find instance ID. Please terminate manually!")
        return False
    
    print(f"Monitoring instance {instance_id} (IP: {ip_address})")
    print(f"Will auto-terminate after {max_runtime_hours} hours or when experiments complete")
    
    last_check = datetime.now()
    
    while True:
        current_time = datetime.now()
        runtime = current_time - start_time
        
        # Check if max runtime exceeded
        if runtime > max_runtime:
            print(f"\n⏰ Max runtime ({max_runtime_hours} hours) exceeded!")
            print("Terminating instance...")
            destroy_instance(instance_id)
            break
        
        # Check if it's time to check status
        if current_time - last_check > check_interval:
            print(f"\n[{current_time.strftime('%H:%M:%S')}] Checking experiment status...")
            
            # Check if results archive exists (indicates completion)
            try:
                result = subprocess.run(
                    ["ssh", "-p", str(port), f"root@{ip_address}", 
                     "ls /workspace/nanda-unfaithful/*.tar.gz 2>/dev/null | wc -l"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    archive_count = int(result.stdout.strip())
                    if archive_count > 0:
                        print("✅ Experiment archives found! Experiments complete.")
                        
                        # Auto-download results before terminating
                        print("Attempting to auto-download results...")
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        local_dir = f"./auto_results_{timestamp}"
                        
                        try:
                            Path(local_dir).mkdir(parents=True, exist_ok=True)
                            download_result = subprocess.run(
                                ["scp", "-P", str(port), 
                                 f"root@{ip_address}:/workspace/nanda-unfaithful/*.tar.gz",
                                 local_dir],
                                timeout=300,
                                capture_output=True,
                                text=True
                            )
                            if download_result.returncode == 0:
                                print(f"✅ Results auto-downloaded to {local_dir}")
                            else:
                                print(f"⚠️ Download failed: {download_result.stderr}")
                                print("Waiting 60 seconds for manual download...")
                                time.sleep(60)
                        except Exception as e:
                            print(f"⚠️ Auto-download error: {e}")
                            print("Waiting 60 seconds for manual download...")
                            time.sleep(60)
                        
                        print("\n🤖 AUTO-TERMINATING instance to prevent overcharging...")
                        if destroy_instance(instance_id):
                            print("💰 Instance terminated! No more charges.")
                        break
                
            except subprocess.TimeoutExpired:
                print("SSH check timed out, instance might be busy")
            except Exception as e:
                print(f"Error checking status: {e}")
            
            last_check = current_time
        
        # Show status
        print(f"\r⏱️  Runtime: {str(runtime).split('.')[0]} / Max: {max_runtime_hours}h | " 
              f"Next check in {int((check_interval - (current_time - last_check)).total_seconds())}s", 
              end='', flush=True)
        
        time.sleep(10)  # Sleep 10 seconds between status updates
    
    print("\n\n✅ Auto-termination complete!")
    return True

def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python start_monitor_and_auto_terminate.py <IP> <PORT> [MAX_HOURS]")
        print("Example: python start_monitor_and_auto_terminate.py 123.45.67.89 12345 4")
        sys.exit(1)
    
    ip = sys.argv[1]
    port = int(sys.argv[2])
    max_hours = float(sys.argv[3]) if len(sys.argv) > 3 else 4
    
    # Check if vastai CLI is installed
    try:
        subprocess.run(["vastai", "--help"], capture_output=True, check=True)
    except:
        print("❌ vastai CLI not found!")
        print("Install with: pip install vastai")
        print("Then login with: vastai set api-key YOUR_API_KEY")
        sys.exit(1)
    
    print("="*50)
    print("VAST.AI AUTO-TERMINATION MONITOR")
    print("="*50)
    print(f"Instance IP: {ip}")
    print(f"SSH Port: {port}")
    print(f"Max Runtime: {max_hours} hours")
    print("="*50)
    print("\nThis script will:")
    print("1. Monitor your experiments")
    print("2. Auto-terminate when complete OR after max runtime")
    print("3. Save you from overcharges!")
    print("\nPress Ctrl+C to cancel auto-termination")
    print("="*50)
    
    try:
        monitor_and_terminate(ip, port, max_hours)
    except KeyboardInterrupt:
        print("\n\n⚠️  Auto-termination cancelled!")
        print("Remember to manually terminate your instance!")
        sys.exit(1)

if __name__ == "__main__":
    main()