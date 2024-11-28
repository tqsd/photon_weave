
def route_operation():
    """
    A decorator, which dynamically calls the
    method of the same name of the object, in
    which the state of the caller resides
    """
    def decorator(method):
        def wrapper(self, *args, **kwargs):
            mn = method.__name__

            # Special cases
            if mn == "resize":
                mn = "resize_fock"
            if self.index is None:
                # State is in the object
                return method(self, *args, **kwargs)

            elif isinstance(self.index, int):
                if not hasattr(self.envelope, mn):
                    raise AttributeError(
                        f"Envelope has no method {mn}"
                        )

                delegate_method = getattr(self.envelope, mn)
                if len(args) > 0:
                    return delegate_method(*args, self, **kwargs)
                else:
                    return delegate_method(self, *kwargs)
            elif isinstance(self.index, tuple):
                if not hasattr(self.composite_envelope, mn):
                    raise AttributeError(
                        f"CompositeEnvelope has no method {mn}"
                        )

                delegate_method = getattr(
                    self.composite_envelope,
                    mn
                    )
                if len(args)>0:
                    return delegate_method(*args, self, **kwargs)
                else:
                    return delegate_method(self, **kwargs)
        return wrapper
    return decorator
