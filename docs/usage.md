## Minimal Example

```python
import pyxations as pyx

# 1) Convert raw files to BIDS
pyx.dataset_to_bids(
    target_folder_path=" Path/to/the/folder/where/the/BIDS/dataset/will/be/created", 
    files_folder_path="Path/to/the/folder/containing/the/EDF/files",  
    dataset_name="dataset_name",
)

# 2) Compute derivatives using REMoDNaV
msg_keywords = ["begin", "end", "press"]
start_msgs   = {"search": ["beginning_of_stimuli"]}
end_msgs     = {"search": ["end_of_stimuli"]}

pyx.compute_derivatives_for_dataset(
    bids_path,
    dataset_format="eyelink",
    detection_algorithm="remodnav",
    msg_keywords=msg_keywords,
    start_msgs=start_msgs,
    end_msgs=end_msgs,
    overwrite=True,
)
```