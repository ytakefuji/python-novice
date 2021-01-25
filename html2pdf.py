# pip install wkhtmltopdf
import pdfkit
import sys
pdfkit.from_url(sys.argv[1], sys.argv[2])
