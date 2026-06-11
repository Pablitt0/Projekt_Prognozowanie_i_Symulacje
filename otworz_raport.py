import os
import subprocess
import pypandoc

script_dir = os.path.dirname(os.path.abspath(__file__))

src = os.path.join(script_dir, "raport.md")
dst = os.path.join(script_dir, "raport.docx")

pypandoc.convert_file(
    src,
    "docx",
    outputfile=dst
)

print("Wygenerowano:", dst)

# otwarcie w Wordzie
os.startfile(dst)
