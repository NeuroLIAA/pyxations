## Minimal Example

```python
import pyxations as pyx

pyx.dataset_to_bids(
    target_folder_path="Path/to/the/folder/where/the/BIDS/dataset/will/be/created", 
    files_folder_path="Path/to/the/folder/containing/the/EDF/files",  
    dataset_name="dataset_name",
)

pyx.compute_derivatives_for_dataset(
    bids_dataset_folder="Path/to/dataset_name",
    msg_keywords=["start_msg", "end_msg"], # multiple msg keywords are allowed
)
```