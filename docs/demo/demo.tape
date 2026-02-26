# VHS tape for SAIA investigate demo, generate with `vhs demo.tape`, start from project root

Output demo.gif

Set Shell "bash"
Set FontSize 14
Set Width 1200
Set Height 300
Set Theme "Dracula"
Set Padding 10
Set PlaybackSpeed 1.3

Hide
Type "export LLM_BASE_URL=http://localhost:18000/v1"
Enter
Sleep 200ms
Show
Type "./examples/investigate.py 'Water boils at 100 degrees Celsius'"
Sleep 650ms
Enter
Sleep 25s
