:: Convert simulate_diffusion.py to simulate_diffusion.ipynb and simulate_diffusion.html
set PYDIR=X:\Python311
del simulate_diffusion.html
%PYDIR%\python.exe -m ipynb_py_convert simulate_diffusion.py simulate_diffusion.ipynb
%PYDIR%\Scripts\jupyter-nbconvert.exe --to=html --ExecutePreprocessor.enabled=True --ExecutePreprocessor.timeout=600 simulate_diffusion.ipynb
:: open HTML
simulate_diffusion.html
pause