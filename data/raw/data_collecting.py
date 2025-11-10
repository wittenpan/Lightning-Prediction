import asyncio
import json
import pandas as pd
import websockets
from pathlib import Path
from collections import deque
import time
"""
10M rows chosen due to potentially future usage of the data for other data heavier models like lstm, transformers, etc. 
"""

# reverse engineering the json from the Blitzortung WebSocket message, json LZW compressed
def lzw_decompress(codes: list) -> str:
    """Decompress LZW-compressed integer codes."""
    dict_size = 256
    dictionary = {i: chr(i) for i in range(dict_size)}
    
    result = []
    prev_code = codes[0]
    result.append(dictionary[prev_code])
    
    for i in range(1, len(codes)):
        code = codes[i]
        
        if code in dictionary:
            entry = dictionary[code]
        elif code == dict_size:
            entry = dictionary[prev_code] + dictionary[prev_code][0]
        else:
            raise ValueError(f"Bad LZW code: {code}")
        
        result.append(entry)
        dictionary[dict_size] = dictionary[prev_code] + entry[0]
        dict_size += 1
        prev_code = code
    
    return ''.join(result)

def decode_message(message: str) -> dict:
    """Decode a Blitzortung WebSocket message."""
    try:
        codes = [ord(c) for c in message]
        decompressed = lzw_decompress(codes)
        return json.loads(decompressed)
    except Exception:
        return None

def get_current_batch_info():
    """Find the most recent batch file and return batch number and existing records."""
    script_dir = Path(__file__).parent
    batch_files = sorted(script_dir.glob('lightning_batch_*.parquet'))
    
    if not batch_files:
        return 0, []
    
    latest_batch = batch_files[-1]
    batch_num = int(latest_batch.stem.split('_')[-1])
    df = pd.read_parquet(latest_batch)
    
    if len(df) < BATCH_SIZE:
        print(f"Resuming batch {batch_num} with {len(df):,} strikes")
        return batch_num, df.to_dict('records')
    else:
        print(f"starting batch {batch_num + 1}")
        return batch_num + 1, []

# configuration
WS_URLS = [f"wss://ws{i}.blitzortung.org/" for i in range(2, 9)]
BATCH_SIZE = 100000
MAX_BATCHES = 100
RECONNECT_DELAY = 3
SAVE_INTERVAL = 10000  # Save progress every 10k strikes (~5-10 strikes/s, takes ~30m)
async def save_batch_async(records, batch_num, is_complete=False):
    """Save batch in a thread pool to avoid blocking."""
    def _save():
        df = pd.DataFrame(records)
        script_dir = Path(__file__).parent
        filename = script_dir / f'lightning_batch_{batch_num:04d}.parquet'
        df.to_parquet(filename, index=False, compression='snappy', engine='pyarrow')
        return filename.name, len(df)  # Return just filename, not full path

    # Run blocking I/O in thread pool
    loop = asyncio.get_event_loop()
    filename, count = await loop.run_in_executor(None, _save)
    
    status = "complete" if is_complete else "progress"
    print(f"Saved {status} batch {batch_num}: {count:,} strikes to {filename}")
    return filename

async def collect_strikes():
    """Main collection function with optimizations."""
    batch_num, records = get_current_batch_info()
    total_strikes = batch_num * BATCH_SIZE + len(records)
    reconnect_count = 0
    ws_index = 0
    
    # performance tracking
    last_print_time = time.time()
    last_save_time = time.time()
    strikes_since_print = 0
    
    while batch_num < MAX_BATCHES:
        ws_url = WS_URLS[ws_index % len(WS_URLS)]
        
        try:
            async with websockets.connect(
                ws_url, 
                ping_interval=20, 
                ping_timeout=10,
                max_size=10000000 #buffer size =10m
            ) as ws:
                if reconnect_count > 0:
                    print(f"Reconnected to {ws_url} (reconnect #{reconnect_count})")
                else:
                    print(f"Connected to {ws_url}")
                
                await ws.send('{"a":111}')
                print(f"Collecting batch {batch_num + 1}/{MAX_BATCHES}")
                
                message_count = 0
                
                async for message in ws:
                    message_count += 1
                    
                    # Decode message
                    data = decode_message(message)
                    
                    if data:
                        total_strikes += 1
                        strikes_since_print += 1
                        records.append(data)
                        
                        current_time = time.time()
                        
                        # print progress every 30 seconds
                        if current_time - last_print_time >= 30.0:
                            elapsed = current_time - last_print_time
                            rate = strikes_since_print / elapsed
                            print(f"Batch {batch_num + 1}: {len(records):,}/{BATCH_SIZE:,} | "
                                  f"Rate: {rate:.1f} strikes/sec | Total: {total_strikes:,}")
                            last_print_time = current_time
                            strikes_since_print = 0
                        
                        # periodic saves to prevent data loss (but less frequent than batch completion)
                        if len(records) % SAVE_INTERVAL == 0 and current_time - last_save_time >= 30:
                            # Non-blocking save, speeds up the collection process
                            asyncio.create_task(save_batch_async(records.copy(), batch_num, False))
                            last_save_time = current_time
                        
                        # complete batch
                        if len(records) >= BATCH_SIZE:
                            # Save asynchronously
                            await save_batch_async(records, batch_num, True)
                            print(f"   Progress: {batch_num + 1}/{MAX_BATCHES} batches "
                                  f"({(batch_num + 1) / MAX_BATCHES * 100:.1f}%)\n")
                            
                            batch_num += 1
                            records = []
                            last_save_time = time.time()
                            
                            if batch_num >= MAX_BATCHES:
                                print(f"\nCollection complete! {MAX_BATCHES} batches "
                                      f"({MAX_BATCHES * BATCH_SIZE:,} total strikes)")
                                return
        
        except Exception as e:
            print(f"\nLost connection: {type(e).__name__}")
            
            # Save progress
            if records:
                await save_batch_async(records, batch_num, False)
            
            reconnect_count += 1
            ws_index += 1
            print(f"Reconnecting in {RECONNECT_DELAY}s... (attempt #{reconnect_count})")
            await asyncio.sleep(RECONNECT_DELAY)
    
    # Save final records
    if records:
        await save_batch_async(records, batch_num, True)
    
    print(f"\nCollection completewith {total_strikes:,} total strikes")

async def main():
    print(f"Target: {MAX_BATCHES} batches × {BATCH_SIZE:,} = {MAX_BATCHES * BATCH_SIZE:,} total\n")
    await collect_strikes()

if __name__ == "__main__":
    asyncio.run(main())