# Executables

Windows double-click launcher:

- [launch_pipeline_gui.bat](/C:/Users/joelb/OneDrive/Vela_partnerships_project/Teacher_student_project/Vela_teacher_student_llm_reasoning/executables/launch_pipeline_gui.bat)

Behavior:

- resolves the repo root relative to the batch file location
- starts the GUI with `C:\Users\joelb\.conda\envs\vela_TRL\python.exe`
- leaves the GUI running in its own window after launch
- pauses only if the expected interpreter or GUI module cannot be found

Validation from a terminal:

```powershell
.\executables\launch_pipeline_gui.bat --check
```
