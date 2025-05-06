import json
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TargetVariable:
    name: str

    @staticmethod
    def from_dict(data: dict) -> 'TargetVariable':
        return TargetVariable(**data)


@dataclass
class Lag:
    name: str
    value: int

    @staticmethod
    def from_dict(data: dict) -> 'Lag':
        # JSON uses "value" as the lag amount
        return Lag(**data)


@dataclass
class FeatureVariable:
    name: str
    include_self: bool
    lag: Optional[List[Lag]] = None

    @staticmethod
    def from_dict(data: dict) -> 'FeatureVariable':
        # If there's a lag list, convert each dict to a Lag
        if 'lag' in data and data['lag'] is not None:
            data['lag'] = [Lag.from_dict(l) for l in data['lag']]
        return FeatureVariable(**data)


@dataclass
class MyConfig:
    name: str
    description: str
    target_variables: List[TargetVariable]
    features: List[FeatureVariable]

    @staticmethod
    def from_dict(data: dict) -> 'MyConfig':
        data['target_variables'] = [
            TargetVariable.from_dict(tv) for tv in data.get('target_variables', [])
        ]
        data['features'] = [
            FeatureVariable.from_dict(f) for f in data.get('features', [])
        ]
        return MyConfig(**data)

    def get_df_names_from_config(self, include_targets=True) -> List[str]:
        """
        Extracts and constructs a list of feature and target variable names based on the
        configuration defined within the object's features and target variables.

        This method iterates through the `features` attribute, checking for lag configurations
        and appending lag names to the list of feature names. Also, if the include_self
        attribute is set to true, then the feature name is added to the list (essentially lag 0).
        It also extracts the names of target variables defined in the `target_variables`
        attribute, combining both sets into a single list.

        :raises AttributeError: If an object in the `features` list or the `target_variables`
            list lacks the expected `lag` or `name` attributes.

        :return: A combined list of names derived from the `lag` configurations of features
            and the names of target variables.
        :rtype: List[str]
        """
        names = []
        for feature in self.features:
            if feature.include_self:
                names.append(feature.name)
            if feature.lag is not None:
                for lag in feature.lag:
                    names.append(lag.name)

        if include_targets:
            targets = [feature.name for feature in self.target_variables]
        else:
            targets = []
        return targets + names


def load_config(path: str) -> MyConfig:
    with open(path, 'r') as file:
        data = json.load(file)
    return MyConfig.from_dict(data)


if __name__ == '__main__':
    # adjust this path as needed
    config = load_config(r'D:\Jetbrains\Python\Projects\solar_irradiance_nowcasting\configs\dni_only\config.json')
    print(config)
    print(config.target_variables)
    print(config.features)
    print(config.features[0].lag)
    print(config.get_df_names_from_config())