import unittest

import torch
from torch import Tensor, nn

from looped_transformer.core.abstractions.typed_module import (
    TypedBranchedModule,
    TypedModule,
)
from looped_transformer.core.modules.branched_module import BranchedModule


class Scale(nn.Module):
    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = nn.Parameter(torch.tensor(factor))

    def forward(self, value: Tensor) -> Tensor:
        return value * self.factor


class ExampleBranchedModule(BranchedModule[Tensor, Tensor]):
    def __init__(self) -> None:
        super().__init__()
        self.first = self._create_branched(
            main_branch=TypedModule(Scale(2.0)),
            side_branches={"triple": TypedModule(Scale(3.0))},
        )
        self.second = self._create_branched(
            main_branch=TypedModule(Scale(5.0)),
            side_branches={"triple": TypedModule(Scale(7.0))},
        )

    def _forward_task(self, value: Tensor) -> Tensor:
        return self.second(self.first(value))


class BranchedModuleTests(unittest.TestCase):
    def test_typed_wrapper_exposes_task_aware_call_and_torch_module(self) -> None:
        module = ExampleBranchedModule()
        model = TypedBranchedModule(module)

        self.assertIs(model.torch, module)
        self.assertEqual(model(torch.tensor(1.0), task="triple").item(), 21.0)
        self.assertEqual(
            model.forward_task(torch.tensor(1.0), task=None).item(),
            10.0,
        )

    def test_assignment_registers_router_under_its_attribute_name(self) -> None:
        model = ExampleBranchedModule()

        self.assertEqual(set(model._modules), {"first", "second"})
        self.assertIn("first.main.factor", model.state_dict())

    def test_none_selects_main_branches_and_name_selects_side_branches(self) -> None:
        model = ExampleBranchedModule()
        value = torch.tensor(1.0)

        self.assertEqual(model(value, None).item(), 10.0)
        self.assertEqual(model(value, "triple").item(), 21.0)

    def test_branches_are_normal_registered_modules_with_gradients(self) -> None:
        model = ExampleBranchedModule()

        model(torch.tensor(1.0), "triple").backward()

        parameters = dict(model.named_parameters())
        self.assertEqual(
            set(parameters),
            {
                "first.main.factor",
                "first.branches.triple.factor",
                "second.main.factor",
                "second.branches.triple.factor",
            },
        )
        self.assertIsNone(parameters["first.main.factor"].grad)
        first_grad = parameters["first.branches.triple.factor"].grad
        second_grad = parameters["second.branches.triple.factor"].grad
        self.assertIsInstance(first_grad, Tensor)
        self.assertIsInstance(second_grad, Tensor)
        assert first_grad is not None
        assert second_grad is not None
        self.assertEqual(first_grad.item(), 7.0)
        self.assertEqual(second_grad.item(), 3.0)

    def test_unknown_task_has_an_actionable_error(self) -> None:
        model = ExampleBranchedModule()

        with self.assertRaisesRegex(
            KeyError, "no branch registered for task 'missing'.*'triple'"
        ):
            model(torch.tensor(1.0), "missing")

        # A failed call must not leak its task into the next invocation.
        self.assertEqual(model(torch.tensor(1.0), None).item(), 10.0)

    def test_task_is_required_and_must_be_the_last_argument(self) -> None:
        model = ExampleBranchedModule()

        with self.assertRaisesRegex(TypeError, "requires at least the task argument"):
            model()
        with self.assertRaisesRegex(TypeError, r"task must be str \| None"):
            model(torch.tensor(1.0), 123)  # type: ignore[arg-type]

    def test_main_state_dict_extracts_one_main_branch(self) -> None:
        model = ExampleBranchedModule()

        state = model.main_state_dict("first")

        self.assertEqual(set(state), {"factor"})
        self.assertEqual(state["factor"].item(), 2.0)
        with self.assertRaisesRegex(KeyError, "no branched router"):
            model.main_state_dict("missing")

    def test_loading_a_main_only_checkpoint_can_ignore_missing_side_branches(
        self,
    ) -> None:
        source = ExampleBranchedModule()
        checkpoint = {
            key: value
            for key, value in source.state_dict().items()
            if ".branches." not in key
        }
        checkpoint["first.main.factor"] = torch.tensor(11.0)
        target = ExampleBranchedModule()

        incompatible = target.load_branched_state_dict(checkpoint)

        self.assertEqual(target(torch.tensor(1.0), None).item(), 55.0)
        self.assertEqual(
            set(incompatible.missing_keys),
            {
                "first.branches.triple.factor",
                "second.branches.triple.factor",
            },
        )
        with self.assertRaisesRegex(RuntimeError, "missing branch keys"):
            target.load_branched_state_dict(checkpoint, strict_branches=True)

    def test_loading_still_rejects_missing_main_parameters_by_default(self) -> None:
        model = ExampleBranchedModule()
        checkpoint = dict(model.state_dict())
        del checkpoint["first.main.factor"]

        with self.assertRaisesRegex(RuntimeError, "missing main keys"):
            model.load_branched_state_dict(checkpoint)


if __name__ == "__main__":
    unittest.main()
