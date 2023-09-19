import re


def get_result(annotator, model, input_text, spelling_threshold, punctuation_threshold):
    """
    Прогоняем через модель каждое предложение между тегами
    """
    html_text = annotator.annotate(input_text)

    current_tags = ''
    tag_is_open = False
    current_text = ''
    result = []
    for s in html_text:
        if s == '<':
            if current_text.strip():  # Проверка, что текст не пустой
                model_res = []
                for t in current_text.split('\n'):
                    if t.strip():
                        model_res.append(
                            model.correct(t, spelling_threshold, punctuation_threshold)
                        )
                result.append('\n'.join(model_res))
            else:
                result.append(current_text)
            current_text = ''
            tag_is_open = True
            current_tags += s
        elif s == '>' and tag_is_open:
            tag_is_open = False
            current_tags += s
            result.append(current_tags)
            current_tags = ''
        elif not tag_is_open:
            current_text += s
        elif tag_is_open:
            current_tags += s
    if current_text.strip():
        result.append(
            model.correct(current_text, spelling_threshold, punctuation_threshold)
        )
    return ''.join(result)


def postprocess_text(text):
    """
    Убираем лишние пробелы и переносы строк
    """
    text = text.replace('<li>;</li>\n', '')
    text = text.replace('<li>;</li>', '')
    return text.strip()
