"""Regression for high-risk markings spreading across PDF paragraph gaps."""
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from documents import extract_document
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


class PdfParagraphTests(unittest.TestCase):
    def test_wrapped_paragraphs_stay_separate_and_code_keeps_lines(self):
        writer = PdfWriter()
        page = writer.add_blank_page(width=595, height=842)
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                                 NameObject('/Subtype'): NameObject('/Type1'),
                                 NameObject('/BaseFont'): NameObject('/Helvetica'),
                                 NameObject('/Encoding'): NameObject('/WinAnsiEncoding')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'):
            DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        paragraphs = [
            ['Measured team observations from the study.',
             'Several outcomes differed between groups.',
             'The original data remains available here.'],
            ['Generated sample for the second paragraph.',
             'This content must stay in another segment.',
             'Do not merge the two unrelated paragraphs.'],
            ['```python', "print('unchanged code lines')", '```'],
        ]
        commands = []
        y = 760
        for lines in paragraphs:
            for text in lines:
                escaped = text.replace('(', '\\(').replace(')', '\\)')
                commands.append(f'BT /F1 12 Tf 40 {y} Td ({escaped}) Tj ET')
                y -= 22
            y -= 26
        stream = DecodedStreamObject()
        stream.set_data('\n'.join(commands).encode('ascii'))
        page[NameObject('/Contents')] = writer._add_object(stream)
        output = io.BytesIO()
        writer.write(output)
        result = extract_document(output.getvalue(), 'wrapped.pdf')
        groups = result['text'].split('\n\n')
        self.assertEqual(len(groups), 3)
        self.assertEqual(groups[0], ' '.join(paragraphs[0]))
        self.assertEqual(groups[1], ' '.join(paragraphs[1]))
        self.assertEqual(groups[2], '\n'.join(paragraphs[2]))
        self.assertEqual(result['sections'][0]['page'], 1)


if __name__ == '__main__':
    unittest.main()
