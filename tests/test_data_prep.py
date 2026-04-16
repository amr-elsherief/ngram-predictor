import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src', 'data_prep'))

from normalizer import Normalizer


def test_lowercase_independent_step():
	normalizer = Normalizer()
	assert normalizer.lowercase("HeLLo WOrLD") == "hello world"


def test_remove_punctuation_independent_step():
	normalizer = Normalizer()
	text = "hello, world! it's great."
	assert normalizer.remove_punctuation(text) == "hello world its great"


def test_remove_numbers_independent_step():
	normalizer = Normalizer()
	text = "abc123 def45 6ghi"
	assert normalizer.remove_numbers(text) == "abc def ghi"


def test_remove_whitespace_independent_step():
	normalizer = Normalizer()
	text = "  hello   world\n\nthis\t is\t\ta test   "
	assert normalizer.remove_whitespace(text) == "hello world this is a test"


def test_normalize_applies_steps_in_sequence():
	normalizer = Normalizer()
	text = "  HELLO,   WoRLD! 123\n\nNew\tLine.  "

	expected = normalizer.remove_whitespace(
		normalizer.remove_numbers(
			normalizer.remove_punctuation(
				normalizer.lowercase(text)
			)
		)
	)

	assert normalizer.normalize(text) == expected
	assert normalizer.normalize(text) == "hello world new line"


def test_strip_gutenberg_removes_header_and_footer_markers():
	normalizer = Normalizer()
	text = (
		"*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
		"Some header metadata\n"
		"Actual book content starts here.\n"
		"It has multiple lines.\n"
		"*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***"
	)

	stripped = normalizer.strip_gutenberg(text)
	assert "START OF THE PROJECT GUTENBERG EBOOK" not in stripped
	assert "END OF THE PROJECT GUTENBERG EBOOK" not in stripped
	assert "Actual book content starts here." in stripped


def test_sentence_tokenize_returns_non_empty_list_for_non_empty_input():
	normalizer = Normalizer()
	text = "This is a sentence. Here is another one!"

	sentences = normalizer.sentence_tokenize(text)

	assert isinstance(sentences, list)
	assert len(sentences) >= 1


def test_word_tokenize_returns_non_empty_string_tokens():
	normalizer = Normalizer()
	sentence = "many   words\twith spaces"

	tokens = normalizer.word_tokenize(sentence)

	assert isinstance(tokens, list)
	assert all(isinstance(token, str) for token in tokens)
	assert all(token != "" for token in tokens)

#example RUN
# python -m pytest tests/test_data_prep.py -q