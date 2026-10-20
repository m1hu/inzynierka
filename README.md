# Praca dyplomowa

## Automatyczna segmentacja zmian nowotworowych na obrazach MRI z wykorzystaniem głębokich sieci splotowych

#### Instrukcja

1. Wymagania
>Windows, Python 3.11 64-bit oraz zgodny sterownik NVIDIA

2. Utworzenie i aktywacja
> `python -m venv .venv`

> `.\.venv\Scripts\Activate.ps1`

>  Jeśli powershell blokuje uruchamianie skryptów wykonać komendę poniżej i ponowić aktywację

> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`

3. Instalacja PyTorch
> `python -m pip install --upgrade pip`

> `python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130`

4. Instalacja bibliotek zapisanych w pliku
> `python -m pip install -r requirements.txt`

5. Rejestracja kernela
> `python -m ipykernel install --user --name inzynierka --display-name "Python (inzynierka)"`

6. Uruchomienie JupyterLab
> `python -m jupyterlab`