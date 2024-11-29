from exa_py import Exa

import os

exa = Exa(os.getenv('exa'))
result = exa.search_and_contents(
  "hottest AI startups",
  type="neural",
  use_autoprompt=True,
  num_results=10,
  text=True,
)

print(len(result.results))

print('result type',type(result))

