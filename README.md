# PyTorch "Super Resolution" example in Streamlit

* [Examples source](https://github.com/pytorch/examples)
* To run: `streamlit run streamlit-enhance.py`

## Issues

* Re-running while PyTorch training is running causes the app to hang forever.
    * Need to shut it down and restart
    * /?restart=1 in s4t
    * PyTorch `Net` hashing requires custom hash_func
        - custom hash_func + custom type + hot-reloading is broken - custom type gets reloaded, and now `(type in hash_funcs) == False`
    * Could we add PyTorch caching? (Can just write to a BytesIO)
        - OR: document this best practice:
        ```python
        bytes = BytesIO()
        torch.save(model, bytes)
        bytes.seek(0)
        bytestring = bytes.getbuffer().tobytes()
        bytes.close()
        # Don't return BytesIO directly, because we mutate it with seeks
        return bytestring
        ```
* s4t
    - Persistent cache unwriteable
    - Force-push hoses things
    - Restarting occasionally hangs forever
 
