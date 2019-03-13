from fastText import FastText

ft = FastText.load_model("lid.176.ftz")
text = 'OpenAI has moved most of its staff to a for-profit LLC. RIP "open-source". RIP "non-profit".'
labels = ft.predict(text)
print(labels)