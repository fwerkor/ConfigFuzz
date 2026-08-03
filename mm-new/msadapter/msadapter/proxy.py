import sys
import types
import importlib
import importlib.metadata
import importlib.abc
import importlib.machinery
from types import ModuleType


class RedirectFinder(importlib.abc.MetaPathFinder):
	def __init__(self, redirect_map):
		self.redirect_map = redirect_map

	def find_spec(self, fullname, path, target=None):
		for proxy_prefix, target_prefix in self.redirect_map.items():
			if fullname == proxy_prefix or fullname.startswith(proxy_prefix + "."):
				target_name = fullname.replace(proxy_prefix, target_prefix, 1)
				try:
					importlib.import_module(target_name)
				except Exception as e:
					raise e

				return importlib.machinery.ModuleSpec(
					name=fullname,
					loader=RedirectLoader(target_name),
					is_package=self._is_package(target_name),
				)
		return None

	def _is_package(self, module_name):
		try:
			module = importlib.import_module(module_name)
			return hasattr(module, "__path__")
		except ImportError:
			return False


class RedirectLoader(importlib.abc.Loader):
	def __init__(self, target_name):
		self.target_name = target_name

	def create_module(self, spec):
		module = ModuleType(spec.name)
		module.__spec__ = spec
		module.__path__ = []
		module.__loader__ = self
		module.__package__ = spec.name
		return module

	def exec_module(self, module):
		class ProxyModule(type(module)):
			import functools

			def __init__(self, *args, **kwargs):
				super().__init__(*args, **kwargs)
				self._proxy_target_name = self.target_name

			@functools.lru_cache(maxsize=None)
			def __getattr__(_, name):
				try:
					target_module = importlib.import_module(self.target_name)
				except ImportError as e:
					raise AttributeError(f"Target module {self.target_name} could not be imported: {e}") from e
				except Exception as e:
					raise e

				if hasattr(target_module, name):
					return getattr(target_module, name)

				try:
					submodule_name = f"{self.target_name}.{name}"
					return importlib.import_module(submodule_name)
				except ImportError as e:
					raise AttributeError(
						f"Module '{self.target_name}' has no attribute '{name}'"
					)

			def __setattr__(_, name, value):
				try:
					target_module = importlib.import_module(self.target_name)
					if not hasattr(target_module, name):
						return
				except Exception as e:
					raise e
				return super().__setattr__(name, value)

			def __reduce__(self):
				"""
                Returns the information needed to reconstruct the object during serialization
                return tuple：(callable, args, state, listitems, dictitems)
                """
				try:
					target_module = importlib.import_module(self.target_name)
					return (self._proxy_reconstruct,
					        (self.__class__.__name__, self.__name__, self.target_name),
					        self.__dict__.copy())
				except ImportError:
					return (self._proxy_reconstruct,
					        (self.__class__.__name__, self.__name__, self.target_name),
					        self.__dict__.copy())

			def __reduce_ex__(self, protocol):
				"""Supports multiple serialization protocols"""
				return self.__reduce__()

			@staticmethod
			def _proxy_reconstruct(class_name, module_name, target_name):
				"""
                Reconstructs proxy modules during deserialization
                """
				try:
					importlib.import_module(target_name)
				except ImportError:
					pass

				spec = importlib.machinery.ModuleSpec(module_name, RedirectLoader(target_name))
				new_module = ModuleType(module_name)
				new_module.__spec__ = spec
				new_module.__name__ = module_name
				new_module.__loader__ = RedirectLoader(target_name)

				proxy_class = type(class_name, (type(new_module),), {
					'target_name': target_name,
					'_proxy_target_name': target_name,
					'__getattr__': ProxyModule.__getattr__,
					'__setattr__': ProxyModule.__setattr__,
					'__reduce__': ProxyModule.__reduce__,
					'__reduce_ex__': ProxyModule.__reduce_ex__,
					'_proxy_reconstruct': ProxyModule._proxy_reconstruct
				})

				new_module.__class__ = proxy_class
				return new_module

		ProxyModule.target_name = self.target_name
		ProxyModule._proxy_target_name = self.target_name

		module.__class__ = ProxyModule


REDIRECT_MAP = {
	"torch": "msadapter",
}


def enable_torch_proxy():
	sys.meta_path.insert(0, RedirectFinder(REDIRECT_MAP))
	import torch
	torch.__version__ = "2.1.1+dev"
	setup_metadata_patch()


def setup_metadata_patch():
	"""fix importlib.metadata can not found torch"""
	orig_distribution = importlib.metadata.distribution
	orig_distributions = importlib.metadata.distributions

	def patched_distribution(dist_name):
		if dist_name == "torch":
			return types.SimpleNamespace(
				version="2.1.1+dev",
				metadata={"Name": "torch", "Version": "2.1.1+dev"},
				read_text=lambda f: f"Name: torch\nVersion: 2.1.1+dev" if f == "METADATA" else None
			)
		return orig_distribution(dist_name)

	def patched_distributions(**kwargs):
		dists = list(orig_distributions(**kwargs))
		dists.append(types.SimpleNamespace(
			name="torch",
			version="2.1.1+dev",
			metadata={"Name": "torch", "Version": "2.1.1+dev"},
			files=[],
			locate_file=lambda p: None,
			_normalized_name='torch',
			entry_points=[]
		))
		return dists

	importlib.metadata.distribution = patched_distribution
	importlib.metadata.distributions = patched_distributions
