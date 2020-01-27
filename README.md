# PyTorch examples in Streamlit

* [Examples source](https://github.com/pytorch/examples)
* To run: `streamlit run streamlit-enhance.py`


## Wishlist

* `st.slider` takes up way too much space
* `st.input`: want to restrict to numeric inputs, use numeric input values
* `st.slider` returns an Array even when we just have a single value
* `st.write`: support for namedtuple
* `st.slider`: always giving float values?
* `st.clear()` as synonym for st.empty (for hiding elements)
* Sometimes I really don't want to rerun immediately after dragging a slider; instead I want to enter a bunch of data and hit GO
    - `st.atomic_widget_group()` or something
    - Something similar to `argparse`
* `st.image_picker` widget (URL, file, etc)
* Copy code icon appears in top-right corner of report when using `st.code` in an `st.cache` function
* I continue to really want to introspect the cache
* When browser reconnects to new server, input values don't get properly updated
* My script is hanging after several minutes and I don't know why, or how to debug it (easily)
* Scriptrunner crash:
    - (I thought I had fixed this one with `_set_execing_flag`, but apparently not!)
    ```
    Exception in thread ScriptRunner.loop:
Traceback (most recent call last):
  File "/usr/local/Cellar/python/3.7.3/Frameworks/Python.framework/Versions/3.7/lib/python3.7/threading.py", line 917, in _bootstrap_inner
    self.run()
  File "/usr/local/Cellar/python/3.7.3/Frameworks/Python.framework/Versions/3.7/lib/python3.7/threading.py", line 865, in run
    self._target(*self._args, **self._kwargs)
  File "/export/streamlit_hackathon_pytorch/venv/lib/python3.7/site-packages/streamlit/ScriptRunner.py", line 328, in _loop
    self._run_script(event_data)
  File "/export/streamlit_hackathon_pytorch/venv/lib/python3.7/site-packages/streamlit/ScriptRunner.py", line 483, in _run_script
    self._run_script(RerunData(argv=None, widget_state=None))
  File "/export/streamlit_hackathon_pytorch/venv/lib/python3.7/site-packages/streamlit/ScriptRunner.py", line 375, in _run_script
    self._set_state(ScriptState.RUNNING)
  File "/export/streamlit_hackathon_pytorch/venv/lib/python3.7/site-packages/streamlit/ScriptRunner.py", line 228, in _set_state
    self.on_state_changed.send(self._state)
  File "/export/streamlit_hackathon_pytorch/venv/lib/python3.7/site-packages/blinker/base.py", line 267, in send
    for receiver in self.receivers_for(sender)]
  File "/export/streamlit_hackathon_pytorch/venv/lib/python3.7/site-packages/blinker/base.py", line 267, in <listcomp>
    for receiver in self.receivers_for(sender)]
  File "/export/streamlit_hackathon_pytorch/venv/lib/python3.7/site-packages/streamlit/ReportContext.py", line 185, in _enqueue_script_state_changed_message
    self._enqueue_new_report_message()
  File "/export/streamlit_hackathon_pytorch/venv/lib/python3.7/site-packages/streamlit/ReportContext.py", line 259, in _enqueue_new_report_message
    self.enqueue(msg)
  File "/export/streamlit_hackathon_pytorch/venv/lib/python3.7/site-packages/streamlit/ReportContext.py", line 147, in enqueue
    self._scriptrunner.maybe_handle_execution_control_request()
  File "/export/streamlit_hackathon_pytorch/venv/lib/python3.7/site-packages/streamlit/ScriptRunner.py", line 297, in maybe_handle_execution_control_request
    raise RerunException()
streamlit.ScriptRunner.RerunException
    ```
* Multiprocessing results in weirdness ("Hogwild" neural net)


## Thoughts

* *Running* the training in streamlit was kinda a pain. *Using* the trained model was much smoother
