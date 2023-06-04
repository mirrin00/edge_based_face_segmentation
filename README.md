# Finding open parts of face using edge-based segmentation

## Install

### With `pip`

```
pip install git+https://github.com/mirrin00/edge_based_face_segmentation@master
```

## Usage

Import package: `from edge_based_face_segmentation import EdgeSegmentor`

Create queues and pass them to `EdgeSegmentor`. Put frames (numpy arrays)
into the input queue and get the result from the output queue. The input
queue receives lists of numpy arrays, the output queue passes lists of
lists with point coordinates. To run segmentor in another thread, use the
`process_queue()` method. To stop processing, put `None`.

Example:
```python
import numpy as np
from edge_based_face_segmentation import EdgeSegmentor
from queue import Queue
import threading

input_queue = Queue()
output_queue = Queue()
edge_seg = EdgeSegmentor(input_queue, output_queue)

seg_thread = threading.Thread(target=edge_seg.process_queue, daemon=True)
seg_thread.start()

input_queue.put([frame1, frame2, ...])  # Put your frames here
result = output_queue.get()

print(result)
input_queue.put(None)
seg_thread.join()
```
