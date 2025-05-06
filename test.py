import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class TargetVariables:
    name: str

    @staticmethod
    def from_dict(data: dict) -> 'TargetVariables':
        return TargetVariables(**data)


@dataclass
class FeatureVariables:
    name: str
    lag: Optional[int] = None

    @staticmethod
    def from_dict(data: dict) -> 'FeatureVariables':
        return FeatureVariables(**data)


@dataclass
class MyConfig:
    name: str
    description: int
    target_variables: list[TargetVariables]
    features: list[FeatureVariables]

    @staticmethod
    def from_dict(data: dict) -> 'MyConfig':
        data['target_variables'] = [TargetVariables.from_dict(tv) for tv in data['target_variables']]
        data['features'] = [FeatureVariables.from_dict(f) for f in data['features']]
        return MyConfig(**data)


with open('config.json', 'r') as file:
    data = json.load(file)

my_config = MyConfig.from_dict(data)

print(my_config)
