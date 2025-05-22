# bewer
Evaluation and analysis framework for automatic speech recognition in Python.

# Tentative Dataset object layout

- [ ] dataset: Dataset
	
	- [ ] plot
		- [ ] {figure_name}: Figure
			- [ ] show: Callable
			- [ ] dump: Callable

	- [ ] error
		- [ ] tags
        - [ ] search

	metrics:	
    - [ ] wer: levenshtein.word.error_rate
    - [ ] cer: levenshtein.char.error_rate
    - [ ] per: span_align.punctuation.error_rate
    - [ ] tags.*: span_align.tags

	- [ ] example: Example		

		- [ ] ref/hyp: Text
			- [ ] words: list[Segment/Word]
				- [ ] length: int
			- [ ] tags.*: list[Segment/Word]
				- [ ] length: int
			- [ ] punctuation: list[Segment]
				- [ ] length: int
		
		- [ ] levenshtein | span_align
			- [ ] word/char | word/tags.*/punctuation
				- [ ] ops: list[Op]
				- [ ] edits: int
				- [ ] error_rate: float
				- [ ] alignment: list[Align]
					- [ ] errors
					- [ ] visualize
						- [ ] print: Callable <- style {inline, inline-st, aligned}
						- [ ] html: Callable <- style {inline, inline-st, aligned}