# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import warnings

from typing import Any, Callable, Literal

import torch
from tensordict import lazy_stack, TensorDict, TensorDictBase
from torch.utils.data import DataLoader
from torchrl.data import Composite, NonTensor

from torchrl.data.llm.chat import History
from torchrl.envs import EnvBase, TransformedEnv

from torchrl.envs.llm.transforms.dataloading import DataLoadingPrimer


class ChatEnv(EnvBase):
    r"""A chat-based environment.

    ChatEnv relies on the :class:`~torchrl.data.llm.History` format to output observations framed as a chat between
    various entities (typically with roles such as `"system"`, `"user"`, `"assistant"` etc.)

    The step function will execute the following operations:

    - Given a prompt (key `"text"`) and an answer string (key `"text_response"`, which is our action), the environment
      will generate a single string that is the concatenation of the two.
    - The text is fed to :meth:`torchrl.data.llm.History.from_text` to produce a full history of the chat so far. This
      should hopefully match the state of the history in the previous step, plus an extra step generated by the new
      action.
    - The last item of that history is then appended to the previous history (we don't replace the history in case
      it contains metadata that cannot be inferred directly from the prompt and response).
    - Optionally, the history is mapped back to a `"text"` entry that can be used to query the LLM in the next round
      of the policy.

    Args:
        batch_size (torch.Size): Expected batch size of the input. Defaults to `(1,)` (null batch sizes such as `()`
            are not recommended as they don't play well with generators).
        system_prompt (str, optional): an optional `"system"` prompt string to use during reset calls.
            Defaults to `None`.
        apply_template (bool, optional): if `True` (and a tokenizer is passed), the history will be parsed to a string
            in the `"text"` entry of the output tensordict at reset time. Defaults to `False`.

            .. note:: If transforms are appended to the environment, the template will be applied to the history before the transform is applied.
                As transforms can encode tools, this means that the text returned by the environment may be incomplete.
                The :class:`~torchrl.modules.llm.vLLMWrapper` and :class:`~torchrl.modules.llm.TransformersWrapper`
                will apply the template to the history when queried if no `"text"` input is provided.

        tokenizer (transformers.PreTrainedTokenizer, *optional*): A tokenizer that will be used to tokenize the text.
            Defaults to `None`.
        template_kwargs (dict[str, any], optional): keyword arguments passed to :meth:`~torchrl.data.llm.History.apply_chat_template`.
            Defaults to `None`.
        system_role (str, optional): the role of the system (at reset time). Defaults to `"system"`.
        user_role (str, optional): the role of the user (at reset time). Defaults to `"user"`.
        make_lazy (bool, optional): if `True`, the environment will return a lazy stack of tensordicts. This is the recommended setting
            for training, since it allows for efficient batching of environment outputs that may have different shapes or contents.
            Defaults to `True`.

    Methods:
        reset (TensorDict): Resets the state of the environment. A tensordict or equivalent with a `"text"` entry must be passed.
        step (TensorDict): Makes a step in the environment (see above for a description of what `step` does).
            A tensordict or equivalent with a `"text"` entry must be passed.

    .. seealso:: To see examples of a `ChatEnv` in action, see :class:`~torchrl.envs.llm.chat.DatasetChatEnv`,
        :class:`~torchrl.envs.llm.GSM8KEnv` and :class:`~torchrl.envs.llm.IFEvalEnv`.

    Examples:
        >>> import pprint
        >>>
        >>> import transformers
        >>> from tensordict import TensorDict, set_list_to_stack
        >>> from torchrl.envs.llm import ChatEnv
        >>> set_list_to_stack(True).set()
        >>>
        >>> tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
        >>>
        >>> env = ChatEnv(batch_size=(1,), tokenizer=tokenizer, apply_template=True, system_prompt="I'm system, do what I want.")
        >>> td_reset = env.reset(TensorDict(text=["I'm the user. I'm going to tell you a little about something."], batch_size=(1,)))
        >>> pprint.pprint(f'{td_reset["history"]=}')
        ('td_reset["history"]=History(\n'
         '    content=NonTensorStack(\n'
         '        [["I\'m system, do what I want.", "I\'m the user. I\'...,\n'
         '        batch_size=torch.Size([1, 2]),\n'
         '        device=None),\n'
         '    role=NonTensorStack(\n'
         "        [['system', 'user']],\n"
         '        batch_size=torch.Size([1, 2]),\n'
         '        device=None),\n'
         '    batch_size=torch.Size([1, 2]),\n'
         '    device=None,\n'
         '    is_shared=False)')
        >>> pprint.pprint(f'{td_reset["text"]=}')
        ('td_reset["text"]=["<|im_start|>system\\nI\'m system, do what I '
         "want.<|im_end|>\\n<|im_start|>user\\nI'm the user. I'm going to tell you a "
         'little about something.<|im_end|>\\n<|im_start|>assistant\\n"]')
        >>> td_action = td_reset.set("text_response", ["This is the action from the assistant!<|im_end|>"])
        >>> td_next = env.step(td_action)
        >>> pprint.pprint(f'{td_next["next", "history"]=}')
        ('td_next["next", "history"]=History(\n'
         '    content=NonTensorStack(\n'
         '        [["I\'m system, do what I want.", "I\'m the user. I\'...,\n'
         '        batch_size=torch.Size([1, 3]),\n'
         '        device=None),\n'
         '    role=NonTensorStack(\n'
         "        [['system', 'user', 'assistant']],\n"
         '        batch_size=torch.Size([1, 3]),\n'
         '        device=None),\n'
         '    batch_size=torch.Size([1, 3]),\n'
         '    device=None,\n'
         '    is_shared=False)')
        >>> pprint.pprint(f'{td_next["next", "text"]=}')
        ('td_next["next", "text"]=["<|im_start|>system\\nI\'m system, do what I '
         "want.<|im_end|>\\n<|im_start|>user\\nI'm the user. I'm going to tell you a "
         'little about something.<|im_end|>\\n<|im_start|>assistant\\nThis is the '
         'action from the assistant!<|im_end|>\\n<|im_start|>assistant\\n"]')

    """

    def __init__(
        self,
        batch_size: tuple | torch.Size | None = None,
        system_prompt: str | None = None,
        apply_template: bool | None = None,
        tokenizer: transformers.AutoTokenizer | None = None,  # noqa: F821
        template_kwargs: dict[str, Any] | None = None,
        system_role: str = "system",
        user_role: str = "user",
        policy_role: str | None = "assistant",
        make_lazy: bool = True,
    ):
        if batch_size is None:
            batch_size = (1,)
        if isinstance(batch_size, int):
            batch_size = (batch_size,)
        if isinstance(batch_size, list):
            batch_size = torch.Size(batch_size)
        if batch_size == ():
            raise ValueError(f"{type(self).__name__} must have at least one dimension")

        super().__init__(batch_size=batch_size)
        self.full_observation_spec = Composite(
            history=History.default_spec(shape=batch_size + (-1,)),
            shape=batch_size,
        )
        self.full_state_spec = self.full_observation_spec.clone()
        self.full_state_spec["text"] = NonTensor(
            shape=self.batch_size, example_data="a string", device=self.device
        )
        self.system_prompt = system_prompt
        self.apply_template = (
            apply_template or (template_kwargs is not None) or (tokenizer is not None)
        )
        self.tokenizer = tokenizer
        if template_kwargs is None:
            template_kwargs = {}
        # FIXME: what to do if True?
        template_kwargs.setdefault("tokenize", False)
        self.template_kwargs = template_kwargs
        if self.apply_template:
            self.full_observation_spec["text"] = NonTensor(
                shape=self.batch_size, example_data="a string", device=self.device
            )
        self.full_action_spec = Composite(
            text_response=NonTensor(
                shape=self.batch_size, example_data="a string", device=self.device
            ),
            batch_size=self.batch_size,
        )
        self.system_role = system_role
        self.user_role = user_role
        self.policy_role = policy_role
        self.make_lazy = make_lazy

    def _step(self, tensordict):
        # Expect action to be a "text_response" string
        action = tensordict["text_response"]
        # Find the total text
        text = tensordict["text"]
        if isinstance(text, str):
            text = [text]
            action = [action]
        text = [t + a for t, a in zip(text, action)]
        # Convert text to a history
        chat_template_name = None
        if self.tokenizer is not None:
            name_or_path = self.tokenizer.name_or_path
            if "qwen" in name_or_path.lower():
                chat_template_name = "qwen"
        parsed_history = History.from_text(text, chat_template_name=chat_template_name)
        # Isolate last element, which should be our action
        local_history = parsed_history[..., -1]
        # Get previous history
        history = tensordict["history"]
        # Check that history has one more item than before
        if history.shape[-1] <= parsed_history.shape[-1]:
            warnings.warn(
                "The parsed history has fewer or the same number than the last element in history."
            )
        if self.policy_role is not None:
            # Iterate over batch and check policy role
            for lh in local_history.unbind(0):
                if lh.role != self.policy_role:
                    raise ValueError(
                        "The role received in the last block parsed from the policy "
                        f"output does not match the expected policy role: received {lh.role} but expected {self.policy_role}.\n"
                        f"Parsed input: {text=}\n"
                        f"Parsed history: {parsed_history=}\n"
                        f"Final element: {local_history=}"
                    )
        # Append history item
        history = history.append(local_history, inplace=False)
        # FIXME: consider done to be always False
        td_out = lazy_stack(
            list(
                TensorDict(
                    history=history,
                    done=torch.zeros(tensordict.shape + (1,), dtype=torch.bool),
                    batch_size=self.batch_size,
                ).unbind(0)
            )
        )
        if self.apply_template:
            td_out["text"] = history.apply_chat_template(
                tokenizer=self.tokenizer, **self.template_kwargs
            )
        return td_out

    def _reset(self, tensordict: TensorDictBase | None):
        if tensordict is None:
            raise RuntimeError(f"{type(self).__name__} expects a tensordict as input")
        # Find the total text
        content = tensordict.get("text")
        if content.batch_size != self.batch_size:
            for s in reversed(self.batch_size):
                content = [content for _ in range(s)]

        # FIXME: Assume the text is not formatted and this is just content
        role = self.user_role
        for s in reversed(self.batch_size):
            role = [role for _ in range(s)]
        history = History(role=role, content=content, batch_size=self.batch_size)
        if self.system_prompt is not None:
            system_role = self.system_role
            history_system = History(
                role=system_role,
                content=self.system_prompt,
            )
            for s in reversed(self.batch_size):
                history_system = lazy_stack([history_system for _ in range(s)])
            history = lazy_stack([history_system, history], -1)
        else:
            history = history.unsqueeze(-1)
        result = TensorDict(
            history=history,
            done=torch.zeros(tensordict.shape + (1,), dtype=torch.bool),
            batch_size=self.batch_size,
        )
        if self.make_lazy:
            result = result.unbind(0)
            result = lazy_stack(list(result), dim=0)
        elif tensordict._lazy:
            result = result.unbind(tensordict.stack_dim)
            result = lazy_stack(list(result), dim=tensordict.stack_dim)
        result.update(tensordict.exclude(*result.keys(True)))
        if self.apply_template:
            template = history.apply_chat_template(
                tokenizer=self.tokenizer, **self.template_kwargs
            )
            result["text"] = template
        return result

    def _set_seed(self, seed):
        return


class DatasetChatEnv(TransformedEnv):
    """Base class for chat environment with queries pulled from a dataset.

    Typical usage include RLHF (Reinforcement Learning from Human feedback) or RLVR (Reinforcement learning with Verifiable rewards).

    Keyword Args:
        dataset (str): The name of the dataset.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to `True`.
        name (str, optional): name of the dataset configuration.
        split (str, optional): the split to use (usually from `"train"`, `"val"` or `"test"`). Defaults to `None` (no split).
        num_envs (int, optional): The number of environments to create. Defaults to `1`.
        repeats (int | None, optional): The number of times to repeat each sample from the dataset (mainly for Monte-Carlo
            based value estimation). If `None`, the dataset is not repeated. Defaults to `None`.
        batch_size_dl (int, optional): The batch size for data loading. Defaults to `1`.
        seed (int | None, optional): The random seed for reproducibility. If `None`, a random seed is used. Defaults to `None`.
        group_repeats (bool, optional): Whether to group repeated samples together. Defaults to `False`.
        tokenizer (transformers.AutoTokenizer | None, optional): The tokenizer to use for text processing. Defaults to `None`.

            .. note:: It is recommended to pass a tokenizer to the environment. This is an easy way to ensure that the
                template applied to the chat history is consistent with the format required by the model.

        device (torch.device | None, optional): The device to use for computations. Defaults to None.
        template_kwargs (dict[str, Any] | None, optional): Additional keyword arguments for the template. Defaults to `None`.
        apply_template (bool | None, optional): Whether to apply the template to the text. Defaults to `False`.
        collate_fn (Callable | None, optional): A custom collate function for data loading. If `None`, a default
            collate function is used. Defaults to `None`.

    .. seealso:: `DatasetChatEnv` is a thin wrapper around :class:`~torchrl.envs.llm.ChatEnv` bucketed with a
        :class:`~torchrl.envs.llm.DataLoadingPrimer` transform. See these two classes for more insight on data format
        and functionality.

    .. seealso:: Examples of `DatasetChatEnv` include :class:`~torchrl.envs.llm.GSM8KEnv` and :class:`~torchrl.envs.llm.IFEvalEnv`.

    """

    SYSTEM_PROMPT: str | None = None

    def __init__(
        self,
        *,
        dataset: str,
        shuffle: bool = True,
        name: str | None = None,
        split: Literal["train", "val", "test"] | None = None,
        num_envs: int = 1,
        repeats: int | None = None,
        batch_size_dl: int = 1,
        seed: int | None = None,
        group_repeats: bool = False,
        tokenizer: transformers.AutoTokenizer | None = None,  # noqa: F821
        device: torch.device | None = None,
        template_kwargs: dict[str, Any] | None = None,
        apply_template: bool | None = False,
        collate_fn: Callable[[Any], Any] | None = None,
    ):
        from datasets import load_dataset
        from tensordict import list_to_stack

        if not list_to_stack():
            raise RuntimeError(
                "list_to_stack() must return True. Use LIST_TO_STACK=1 or `tensordict.set_list_to_stack(True).set()` "
                "at the beginning of the script."
            )

        batch_size = (num_envs,)

        dataset = load_dataset(dataset, name)
        if split is None and "train" in dataset:
            split = "train"
        if split is not None:
            dataset = dataset[split]
        # Env
        if seed is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator(device=torch.get_default_device())
        generator.manual_seed(seed)

        dataloader = DataLoader(  # noqa: TOR401
            dataset,
            batch_size=batch_size_dl,
            shuffle=shuffle,
            collate_fn=collate_fn,
            generator=generator,
        )

        primer = DataLoadingPrimer(
            dataloader=dataloader,
            repeats=repeats,
            device=device,
            group_repeats=group_repeats,
            batch_size=batch_size,
        )
        env_base = ChatEnv(
            batch_size=batch_size,
            system_prompt=self.SYSTEM_PROMPT,
            tokenizer=tokenizer,
            template_kwargs=template_kwargs,
            apply_template=apply_template,
        )
        return super().__init__(env_base, primer)

    def reset_dataloader(self):
        """Reset the dataloader.

        This is useful when the dataloader is not infinite and we want to reset it.

        Returns:
            self: The environment itself.
        """
        self.transform[0].reset_dataloader()
        return self
