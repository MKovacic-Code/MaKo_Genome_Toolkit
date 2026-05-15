import os
import subprocess
from pathlib import Path

def create_shortcut():
    root_dir = Path(__file__).resolve().parent.parent
    venv_pythonw = root_dir / ".venv" / "Scripts" / "pythonw.exe"
    script_path = root_dir / "nucleotide_gui.pyw"
    icon_path = root_dir / "assets" / "backpack_icon.ico"
    shortcut_name = "MaKo Genome Toolkit.lnk"
    
    desktop = Path(os.path.expanduser("~")) / "Desktop"
    shortcut_path = desktop / shortcut_name
    
    if not venv_pythonw.exists():
        venv_pythonw = "pythonw.exe"
    
    ps_command = f"""
    $WshShell = New-Object -ComObject WScript.Shell
    $Shortcut = $WshShell.CreateShortcut('{shortcut_path}')
    $Shortcut.TargetPath = '{venv_pythonw}'
    $Shortcut.Arguments = '"{script_path}"'
    $Shortcut.WorkingDirectory = '{root_dir}'
    $Shortcut.IconLocation = '{icon_path}'
    $Shortcut.Save()
    """
    
    try:
        subprocess.run(["powershell", "-Command", ps_command], check=True)
        print(f"Created desktop shortcut: {shortcut_path}")
        
        local_shortcut = root_dir / shortcut_name
        ps_command_local = ps_command.replace(str(shortcut_path), str(local_shortcut))
        subprocess.run(["powershell", "-Command", ps_command_local], check=True)
        print(f"Created local shortcut: {local_shortcut}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    create_shortcut()
