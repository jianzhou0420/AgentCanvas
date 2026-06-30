"""LMP — Language Model Program (vendored from VoxPoser, Huang et al. 2023).

Upstream: https://github.com/huangwl18/VoxPoser  src/LMP.py

Adapted for AgentCanvas:
  * `_cached_api_call` now routes through litellm using `app.llm.call.get_llm_config()`
    instead of calling `openai.ChatCompletion.create` directly. Cache shape is
    preserved (key = kwargs dict containing `messages`).
  * DiskCache stores under `~/.cache/agentcanvas/voxposer_llm/` (per-user).
  * Retry loop on RateLimitError/APIConnectionError dropped — litellm handles
    retries internally via `num_retries`.
  * Legacy completion endpoint (gpt-3.5-turbo-instruct) removed; chat-only.
"""

import os
import time
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
from .utils import load_prompt, DynamicObservation, IterableDynamicObservation
from .LLM_cache import DiskCache


_DEFAULT_CACHE_DIR = os.path.expanduser('~/.cache/agentcanvas/voxposer_llm')


class LMP:
    """Language Model Program (LMP), adopted from Code as Policies."""
    def __init__(self, name, cfg, fixed_vars, variable_vars, debug=False, env='rlbench'):
        self._name = name
        self._cfg = cfg
        self._debug = debug
        self._base_prompt = load_prompt(f"{env}/{self._cfg['prompt_fname']}.txt")
        self._stop_tokens = list(self._cfg['stop'])
        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        self.exec_hist = ''
        self._context = None
        self._cache = DiskCache(
            cache_dir=_DEFAULT_CACHE_DIR,
            load_cache=self._cfg['load_cache'],
        )

    def clear_exec_hist(self):
        self.exec_hist = ''

    def build_prompt(self, query):
        if len(self._variable_vars) > 0:
            variable_vars_imports_str = f"from utils import {', '.join(self._variable_vars.keys())}"
        else:
            variable_vars_imports_str = ''
        prompt = self._base_prompt.replace('{variable_vars_imports}', variable_vars_imports_str)

        if self._cfg['maintain_session'] and self.exec_hist != '':
            prompt += f'\n{self.exec_hist}'

        prompt += '\n'  # separate prompted examples with the query part

        if self._cfg['include_context']:
            assert self._context is not None, 'context is None'
            prompt += f'\n{self._context}'

        user_query = f'{self._cfg["query_prefix"]}{query}{self._cfg["query_suffix"]}'
        prompt += f'\n{user_query}'

        return prompt, user_query

    def _cached_api_call(self, **kwargs):
        """Call LLM via litellm with disk cache.

        Mirrors upstream's chat-mode prompt-to-messages rewrite, then routes
        through AgentCanvas's `get_llm_config()` to pick api_key / base_url /
        provider prefix from the active profile (or AGENTCANVAS_* env vars).
        Cache key is the post-rewrite kwargs (containing `messages`), matching
        upstream cache shape.
        """
        # Chat-mode prompt rewrite — only path we ship.
        user1 = kwargs.pop('prompt', None)
        if user1 is not None:
            new_query = '# Query:' + user1.split('# Query:')[-1]
            user1 = ''.join(user1.split('# Query:')[:-1]).strip()
            user1 = (
                "I would like you to help me write Python code to control a robot arm "
                "operating in a tabletop environment. Please complete the code every time "
                "when I give you new query. Pay attention to appeared patterns in the given "
                "context code. Be thorough and thoughtful in your code. Do not include any "
                "import statement. Do not repeat my question. Do not provide any text "
                "explanation (comment in code is okay). I will first give you the context "
                f"of the code below:\n\n```\n{user1}\n```\n\nNote that x is back to "
                "front, y is left to right, and z is bottom to up."
            )
            assistant1 = 'Got it. I will complete what you give me next.'
            user2 = new_query
            if user1.split('\n')[-4].startswith('objects = ['):
                obj_context = user1.split('\n')[-4]
                user1 = '\n'.join(user1.split('\n')[:-4]) + '\n' + '\n'.join(user1.split('\n')[-3:])
                user2 = obj_context.strip() + '\n' + user2
            kwargs['messages'] = [
                {"role": "system", "content": "You are a helpful assistant that pays attention to the user's instructions and writes good python code for operating a robot arm in a tabletop environment."},
                {"role": "user", "content": user1},
                {"role": "assistant", "content": assistant1},
                {"role": "user", "content": user2},
            ]

        if kwargs in self._cache:
            print('(using cache)', end=' ')
            return self._cache[kwargs]

        # Lazy-import litellm + AgentCanvas LLM config — keeps module
        # importable in environments without the backend on PYTHONPATH.
        import litellm
        from app.llm.call import get_llm_config

        cfg = get_llm_config()
        if cfg is None:
            raise RuntimeError(
                "VoxPoser LMP: no LLM profile or env config found. "
                "Set AGENTCANVAS_API_KEY + AGENTCANVAS_MODEL or activate a profile."
            )
        requested_model = kwargs.get('model', cfg.model)
        litellm_model = f"{cfg.litellm_prefix}/{requested_model}"

        response = litellm.completion(
            model=litellm_model,
            messages=kwargs['messages'],
            max_tokens=kwargs.get('max_tokens', 512),
            temperature=kwargs.get('temperature', 0),
            stop=kwargs.get('stop') or None,
            api_key=cfg.api_key or None,
            api_base=cfg.base_url or None,
            timeout=60.0,
            num_retries=2,
        )
        ret = response.choices[0].message.content
        ret = ret.replace('```', '').replace('python', '').strip()
        self._cache[kwargs] = ret
        return ret

    def __call__(self, query, **kwargs):
        prompt, user_query = self.build_prompt(query)

        start_time = time.time()
        code_str = self._cached_api_call(
            prompt=prompt,
            stop=self._stop_tokens,
            temperature=self._cfg['temperature'],
            model=self._cfg['model'],
            max_tokens=self._cfg['max_tokens'],
        )
        print(f'*** LLM call took {time.time() - start_time:.2f}s ***')

        if self._cfg['include_context']:
            assert self._context is not None, 'context is None'
            to_exec = f'{self._context}\n{code_str}'
            to_log = f'{self._context}\n{user_query}\n{code_str}'
        else:
            to_exec = code_str
            to_log = f'{user_query}\n{to_exec}'

        to_log_pretty = highlight(to_log, PythonLexer(), TerminalFormatter())

        if self._cfg['include_context']:
            print('#'*40 + f'\n## "{self._name}" generated code\n' + f'## context: "{self._context}"\n' + '#'*40 + f'\n{to_log_pretty}\n')
        else:
            print('#'*40 + f'\n## "{self._name}" generated code\n' + '#'*40 + f'\n{to_log_pretty}\n')

        gvars = merge_dicts([self._fixed_vars, self._variable_vars])
        lvars = kwargs

        # return function instead of executing it so we can replan using latest obs (do not do this for high-level UIs)
        if not self._name in ['composer', 'planner']:
            to_exec = 'def ret_val():\n' + to_exec.replace('ret_val = ', 'return ')
            to_exec = to_exec.replace('\n', '\n    ')

        if self._debug:
            # only "execute" function performs actions in environment, so we comment it out
            action_str = ['execute(']
            for s in action_str:
                # Upstream caught the exec error and dropped into pdb here;
                # under canvas we let it propagate so the engine node fails
                # cleanly via its `error` output port.
                exec_safe(to_exec.replace(s, f'# {s}'), gvars, lvars)
        else:
            exec_safe(to_exec, gvars, lvars)

        self.exec_hist += f'\n{to_log.strip()}'

        if self._cfg['maintain_session']:
            self._variable_vars.update(lvars)

        if self._cfg['has_return']:
            if self._name == 'parse_query_obj':
                try:
                    # there may be multiple objects returned, but we also want them to be unevaluated functions so that we can access latest obs
                    return IterableDynamicObservation(lvars[self._cfg['return_val_name']])
                except AssertionError:
                    return DynamicObservation(lvars[self._cfg['return_val_name']])
            return lvars[self._cfg['return_val_name']]


def merge_dicts(dicts):
    return {
        k : v
        for d in dicts
        for k, v in d.items()
    }


def exec_safe(code_str, gvars=None, lvars=None):
    banned_phrases = ['import', '__']
    for phrase in banned_phrases:
        assert phrase not in code_str

    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([
        gvars,
        {'exec': empty_fn, 'eval': empty_fn}
    ])
    try:
        exec(code_str, custom_gvars, lvars)
    except Exception as e:
        import traceback as _tb
        print(f'Error executing code:\n{code_str}')
        print(f'Exception: {type(e).__name__}: {e}')
        print('Traceback:')
        _tb.print_exc()
        raise e
