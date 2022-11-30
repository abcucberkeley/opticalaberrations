import tensorflow as tf
import time

def test_tf():

    print(f"\nfinding physical devices")    
    physical_devices = tf.config.list_physical_devices('GPU') # runtime not initialized.

    print(f"\ninitializing physical devices")    
    physical_devices = tf.config.list_logical_devices('GPU') # runtime initialized.
    print("Num GPUs:", len(physical_devices))
    
    # time.sleep(10)
    assert len(physical_devices) > 0
    return f"Num of GPUs that tensorflow found:  {len(physical_devices)} : {physical_devices}"

