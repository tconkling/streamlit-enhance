# PyTorch examples in Streamlit

* [Examples source](https://github.com/pytorch/examples)
* To run: `streamlit run streamlit-enhance.py`

## Issues

* Re-running while PyTorch training is running causes the app to hang forever.
    * Need to shut it down and restart
    * /?restart=1 in s4t
    * PyTorch `Net` hashing requires custom hash_func
        - custom hash_func + custom type + hot-reloading is broken - custom type gets reloaded, and now `(type in hash_funcs) == False`
    * Could we add PyTorch caching? (Can just write to a BytesIO)
