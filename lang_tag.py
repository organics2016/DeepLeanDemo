import re


def add_language_tags(input_string):
    # Define the regular expression patterns for each language
    en_pattern = r'[\x00-\x7F]+'  # ASCII characters
    zh_pattern = r'[\u4e00-\u9fff]+'  # Chinese characters
    ja_pattern = r'[\u3040-\u30ff]+'  # Hiragana, Katakana, and Kanji characters

    # Compile the regular expressions
    en_regex = re.compile(en_pattern)
    zh_regex = re.compile(zh_pattern)
    ja_regex = re.compile(ja_pattern)

    # Find all matches for each language
    en_matches = en_regex.findall(input_string)
    zh_matches = zh_regex.findall(input_string)
    ja_matches = ja_regex.findall(input_string)

    # Add language tags to the matches
    en_tagged = [f'[en]{match}[en]' for match in en_matches]
    zh_tagged = [f'[zh]{match}[zh]' for match in zh_matches]
    ja_tagged = [f'[ja]{match}[ja]' for match in ja_matches]

    # Combine the tagged matches into a single string
    output_string = ''
    for match in en_tagged + zh_tagged + ja_tagged:
        output_string += match

    # Return the output string
    return output_string


res = add_language_tags("    Regular expression,. 则表达式，あアいイうウえエおオ。323123322")
print(res)
