from PyPDF2 import PdfFileMerger
pdfs = ['split2.pdf','split3.pdf']

merger = PdfFileMerger()
for pdf in pdfs:
    merger.append(pdf)

merger.write("result.pdf")
merger.close()
