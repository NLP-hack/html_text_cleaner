import re


class HTMLAnnotator:

    @staticmethod
    def _base_formatting(text):
        text = re.sub(r'(?<=\n)"|(?<=\n)\n*|&nbsp', r'', text)
        text = re.sub(r':((?=(\d\.))|(?!(\n| |\d)))', r':\n', text)
        text = re.sub(r';(?!\n)', r';\n', text)
        text = re.sub(r'(\d|[а-я]|[^\w\s"«])([А-Я])', r'\1 \2', text)
        return text

    @staticmethod
    def _make_list_titles(text):
        text = re.sub(r'(.*?:(?=\n))', r'<p><b>\1</b></p>', text)
        return text

    @staticmethod
    def _make_lists(text):
        text = re.sub(r'(.*?;)', r'  <li>\1</li>', text)
        text = re.sub(r'(?<=(</p>\n))([\s\S]*)(?=(<p>)*)', r'<ul>\n\2\n</ul>', text)
        text = re.sub(r'(?<=(</li>\n))(.*)(?=(\n</ul>))', r'  <li>\2</li>', text)
        return text

    @staticmethod
    def _make_paragraphs(text):
        text = re.sub(r'(?<=\n)([^<]*)(?=\n)', r'<p>\1</p>', text)
        return text

    @staticmethod
    def _post_processing(text):
        text = re.sub(r'<p></p>', r'', text)
        return text

    def annotate(self, text):
        text = self._base_formatting(text)
        text = self._make_list_titles(text)
        text = self._make_lists(text)
        text = self._make_paragraphs(text)
        text = self._post_processing(text)
        return text
