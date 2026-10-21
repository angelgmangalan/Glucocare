"""Safe errors raised by the food-image workflow."""


class FoodVisionError(RuntimeError):
    """A user-safe image analysis or provider error."""
