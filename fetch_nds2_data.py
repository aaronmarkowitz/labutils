import nds2
import numpy as np
import subprocess
import glob
import os

# Set parameters
server = '192.168.1.11' # Replace with your NDS2 server
port = 8088              # Default NDS2 port
channel = 'Y1:DMT-LESZ_PIT_MON_OUT_DQ'   # Replace with your channel name
start_time = 1434745212          # GPS start time
duration = 60                    # Duration in seconds

# Connect to NDS2 server
conn = nds2.connection(server, port)

def check_channel_in_frames(channel, gps_start, gps_end, frame_dir="/frames/full", frame_prefix="Y-R-"):
    """
    Check if the channel is present in any raw frame files covering the requested GPS interval.
    """
    # Calculate the subdirectory names based on GPS times
    start_subdir = str(gps_start)[:5]
    end_subdir = str(gps_end)[:5]
    
    # Generate a list of all possible subdirectory names in the range
    subdir_range = range(int(start_subdir), int(end_subdir) + 1)
    subdirs = [str(subdir) for subdir in subdir_range]
    
    print(f"Searching for frame files with prefix '{frame_prefix}' in GPS subdirectories {start_subdir}-{end_subdir}")
    
    # Build a list of frame files from the appropriate subdirectories
    frame_files = []
    for subdir in subdirs:
        subdir_path = os.path.join(frame_dir, subdir)
        if os.path.exists(subdir_path):
            subdir_files = sorted(glob.glob(os.path.join(subdir_path, f"{frame_prefix}*.gwf")))
            frame_files.extend(subdir_files)
    
    print(f"Found {len(frame_files)} frame files in relevant GPS subdirectories.")
    if frame_files:
        print("Sample frame file paths:")
        for f in frame_files[:5]:
            print(f"  {f}")
    else:
        # Diagnostic: print subdirectory structure and sample files
        print(f"No frame files found in subdirectories {start_subdir}-{end_subdir}.")
        print("Checking if subdirectories exist:")
        for subdir in subdirs:
            subdir_path = os.path.join(frame_dir, subdir)
            if os.path.exists(subdir_path):
                print(f"  Directory {subdir_path} exists")
                # Show sample of what's in this directory
                files = os.listdir(subdir_path)[:5] if os.listdir(subdir_path) else []
                print(f"    Contains files: {files}")
            else:
                print(f"  Directory {subdir_path} does not exist")

    found = False
    checked_files = 0
    for frame_file in frame_files:
        # Optionally, filter by GPS time in filename for efficiency
        # Example filename: Y-R-14347452-16.gwf
        try:
            basename = os.path.basename(frame_file)
            parts = basename.split('-')
            if len(parts) >= 4:
                frame_gps = int(parts[2])
                frame_len = int(parts[3].split('.')[0])
                if (frame_gps + frame_len) < gps_start or frame_gps > gps_end:
                    continue  # Skip frames outside interval
        except Exception:
            pass  # If filename parsing fails, just try the file

        checked_files += 1
        # Use FrChannels to check for channel presence
        try:
            result = subprocess.run(
                ["FrChannels", frame_file],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10
            )
            if channel in result.stdout:
                print(f"Channel '{channel}' FOUND in frame file: {frame_file}")
                found = True
                break
        except Exception as e:
            print(f"Error running FrChannels on {frame_file}: {e}")

    if not found:
        print(f"Channel '{channel}' NOT found in any checked frame files ({checked_files} files).")
        print("You may also use FrDump for a more detailed inspection, e.g.:")
        print(f"  FrDump {frame_files[0]} | grep {channel}" if frame_files else "No frame files found.")

def get_nds2_server_info(connection):
    """
    Query the NDS2 server for information about its configuration.
    """
    print("\n=== NDS2 Server Information ===")
    try:
        # Get NDS server version
        print(f"NDS Server: {connection.host}:{connection.port}")
        print(f"Server version: {connection.nds}")
        
        # Get available channels
        print(f"Fetching available channels from NDS server...")
        available_channels = connection.find_channels("*", connection.unique_match)
        print(f"Total channels available: {len(available_channels)}")
        if available_channels:
            print("Sample channels:")
            for ch in available_channels[:5]:
                print(f"  {ch.name}")
        
        # Try to get server paths information
        print("\nAttempting to query server data paths...")
        try:
            # Different ways to potentially get path info
            server_info = connection.recv_response("INFO")
            print(f"Server INFO response: {server_info}")
        except Exception as e:
            print(f"Could not get server INFO: {e}")
            
        return available_channels
    except Exception as e:
        print(f"Error getting NDS2 server information: {e}")
        return []

def list_channels_with_frdump(frame_dir="/frames/full", frame_prefix="Y-R-", gps_time=None):
    """
    Use FrDump to list all channels in the most recent frame file.
    If gps_time is provided, look in the appropriate GPS subdirectory.
    """
    frame_files = []
    
    if gps_time:
        # Use the GPS subdirectory approach
        gps_subdir = str(gps_time)[:5]
        subdir_path = os.path.join(frame_dir, gps_subdir)
        print(f"Searching for frame files in GPS subdirectory: {subdir_path}")
        
        # Check if directory exists and is accessible
        if os.path.exists(subdir_path):
            print(f"Directory {subdir_path} exists. Checking permissions:")
            try:
                if os.access(subdir_path, os.R_OK):
                    print(f"Directory {subdir_path} is readable")
                    frame_files = sorted(glob.glob(os.path.join(subdir_path, f"{frame_prefix}*.gwf")))
                else:
                    print(f"Directory {subdir_path} is not readable. Check permissions.")
                    # Try to show who owns the directory
                    try:
                        import pwd, grp
                        stats = os.stat(subdir_path)
                        owner = pwd.getpwuid(stats.st_uid).pw_name
                        group = grp.getgrgid(stats.st_gid).gr_name
                        print(f"Directory owned by: {owner}:{group}, mode: {oct(stats.st_mode)}")
                        print(f"Current user: {os.getenv('USER') or os.getlogin()}")
                    except Exception as e:
                        print(f"Could not determine directory ownership: {e}")
            except Exception as e:
                print(f"Error accessing directory {subdir_path}: {e}")
        else:
            print(f"Directory {subdir_path} does not exist.")
            # Try to find out why - check if the path exists on the server side
            print("\nNDS2 servers typically use internal paths to access frame files which")
            print("might differ from what's accessible on your client machine.")
            print("This suggests the NDS2 server accesses frames via a path that's")
            print("not directly accessible from your current environment.")
            
            # Alternative paths to try
            alt_paths = [
                "/frames/Y1/full", 
                f"/frames/{gps_subdir}",
                f"/data/frames/full/{gps_subdir}", 
                f"/opt/rtcds/yor/y1/frames/full/{gps_subdir}"
            ]
            
            print("\nTrying alternative paths that might contain frame files:")
            for path in alt_paths:
                if os.path.exists(path):
                    print(f"Found alternative path: {path}")
                    alt_files = sorted(glob.glob(os.path.join(path, f"{frame_prefix}*.gwf")))
                    if alt_files:
                        print(f"  Contains {len(alt_files)} frame files with prefix '{frame_prefix}'")
                        frame_files.extend(alt_files)
                        break
            
            # Try to get server paths from ligo-confmanagerdb config
            try:
                print("\nAttempting to check LIGO configuration for frame paths:")
                result = subprocess.run(
                    ["ligo-confmanagerdb", "config", "get", "-n", "nds2"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10
                )
                if "frame_dir" in result.stdout:
                    print(f"Found NDS2 frame_dir in config: {result.stdout}")
            except Exception as e:
                print(f"Could not check LIGO configuration: {e}")
    else:
        # Fall back to original recursive glob approach
        print(f"Searching for frame files with prefix '{frame_prefix}' in directory: {frame_dir}")
        frame_files = sorted(glob.glob(os.path.join(frame_dir, "**", f"{frame_prefix}*.gwf"), recursive=True))
    
    print(f"Found {len(frame_files)} frame files.")
    if frame_files:
        # Diagnostic: print subdirectory structure and sample files
        print("Sample frame file paths:")
        for f in frame_files[:5]:
            print(f"  {f}")
    else:
        print("No frame files found.")

    # For the found frame files, use FrDump to list channels
    found_channels = set()
    try:
        for frame_file in frame_files:
            try:
                result = subprocess.run(
                    ["FrDump", frame_file],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20
                )
                # FrDump output: look for lines starting with "chan:"
                for line in result.stdout.splitlines():
                    if line.strip().startswith("chan:"):
                        # Example: chan: Y1:DMT-LESZ_PIT_MON_OUT_DQ
                        found_channels.add(line.strip().split("chan:")[1].strip())
            except Exception as e:
                print(f"Error running FrDump on {frame_file}: {e}")
        
        if found_channels:
            print("Channels found in the most recent frame:")
            for ch in found_channels:
                print("  " + ch)
        else:
            print("No channels found in the most recent frame file.")
    except Exception as e:
        print(f"Error processing frame files with FrDump: {e}")
            
# Request data
try:
    buffers = conn.fetch(start_time, start_time + duration, [channel])
    if buffers:
        buf = buffers[0]
        data = buf.data
        actual_start = buf.gps_seconds
        actual_duration = len(data) / buf.sample_rate
        actual_end = actual_start + actual_duration
        times = np.arange(actual_start, actual_end, 1.0/buf.sample_rate)
        print(f"Fetched {len(data)} samples for {channel}")
        print(f"Data covers GPS {actual_start} to {actual_end}")
        if actual_start > start_time or actual_end < (start_time + duration):
            print("Warning: Data does not fully cover requested interval. There may be gaps.")
    else:
        print("No data returned. All requested data is missing.")
        print("Possible reasons:")
        print("- Channel was not being acquired during this time.")
        print("- There is a gap in the raw frames for this channel.")
        print("- The channel name or server is incorrect.")
except Exception as e:
    print(f"Error fetching data: {e}")
    error_str = str(e).lower()
    
    if "invalid channel name" in error_str:
        print("\nThe channel name appears to be invalid according to the NDS2 server.")
        print(f"Requested channel: '{channel}'")
        
        # Verify if channel exists through a different method
        print("\nChecking if channel exists in NDS2 server's channel list...")
        try:
            all_channels = conn.find_channels(channel, conn.unique_match)
            if all_channels:
                print(f"FOUND! Channel '{channel}' exists on the server.")
                print("This is strange - it exists but fetch() can't retrieve it.")
                print("Possible causes:")
                print("- The channel might exist now but not for the requested time period")
                print("- There might be a permissions issue accessing this channel's data")
                print("- The channel's data type might be incompatible with the request")
            else:
                print(f"NOT FOUND! Channel '{channel}' does not exist on the server.")
                
                # Find similar channel names to help with typos
                print("\nLooking for similar channel names:")
                similar = find_similar_channels(conn, channel)
                if similar:
                    print("Similar channels found that might be what you're looking for:")
                    for ch in similar:
                        print(f"  {ch}")
                    print("\nTry one of these channels instead.")
                else:
                    print("No similar channels found.")
                
                print("\nTo list all available channels:")
                print(f"  import nds2; conn = nds2.connection('{server}', {port})")
                print("  channels = conn.find_channels('*', conn.unique_match)")
                print("  [ch.name for ch in channels if 'KEYWORD' in ch.name]")
        except Exception as ex:
            print(f"Error checking channel existence: {ex}")
    
    elif "gap" in error_str:
        print("\nIt appears there is a gap in the data for the requested interval.")
        print("Suggested actions:")
        print("- Check if the DAQ was running and acquiring this channel at the requested time.")
        print("- Use FrChannels or FrDump to inspect frame files for channel presence.")
        print("- Verify the DAQ block/channel list in your model includes this channel.")
        print("- Confirm the NDS2 server has access to the relevant raw frames.")
        
        # Query NDS2 server for information
        print("\nQuerying NDS2 server for available channels and configuration...")
        available_channels = get_nds2_server_info(conn)
        
        # Check if our channel is among the available ones
        channel_exists = any(ch.name == channel for ch in available_channels) if available_channels else False
        if channel_exists:
            print(f"\nThe channel '{channel}' IS available on the NDS2 server.")
            print("This suggests the channel exists but data is missing for the requested timespan.")
        else:
            print(f"\nThe channel '{channel}' is NOT found among available NDS2 channels.")
            print("This suggests the channel doesn't exist or is not being served by NDS2.")
        
        # Continue with the rest of the diagnostics
        print("\nChecking frame files for channel presence...")
        
        # Attempt to diagnose directory/permission issues
        print("\nDiagnostic information about frame directories:")
        frame_dir = "/frames/full"
        gps_subdir = str(start_time)[:5]
        subdir_path = os.path.join(frame_dir, gps_subdir)
        
        print(f"Checking path: {subdir_path}")
        # Check basic path existence
        if os.path.exists(frame_dir):
            print(f"Base directory {frame_dir} exists")
            # List some contents to verify
            try:
                contents = sorted(os.listdir(frame_dir))[:10]
                print(f"Sample contents of {frame_dir}: {contents}")
            except Exception as perm_err:
                print(f"Permission error reading {frame_dir}: {perm_err}")
        else:
            print(f"Base directory {frame_dir} does not exist or is not accessible")
            
        # Try alternative paths
        alt_paths = ["/frames", "/home/frames/full", "/data/frames", "/frames/Y1"]
        for path in alt_paths:
            if os.path.exists(path):
                print(f"Alternative path {path} exists. Contents:")
                try:
                    print(f"  {sorted(os.listdir(path))[:5]}")
                except Exception as alt_err:
                    print(f"  Error reading {path}: {alt_err}")
        
        # Continue with regular checks
        check_channel_in_frames(channel, start_time, start_time + duration, frame_dir="/frames/full")
        list_channels_with_frdump(frame_dir="/frames/full", gps_time=start_time)
