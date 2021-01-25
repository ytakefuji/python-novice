import os,sys
from PyPDF2 import PdfFileReader, PdfFileWriter

pdf = PdfFileReader(sys.argv[1])
for page in range(pdf.getNumPages()):
    pdf_writer = PdfFileWriter()
    pdf_writer.addPage(pdf.getPage(page))

    output_filename = '{}{}.pdf'.format('split', page+1)

    with open(output_filename, 'wb') as out:
        pdf_writer.write(out)

    print('Created: {}'.format(output_filename))
