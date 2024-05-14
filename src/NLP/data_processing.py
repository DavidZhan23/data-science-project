import pandas as pd
import re

def load_data(filepath):
    requirements = pd.read_csv(filepath)
    req = pd.DataFrame(requirements, columns=['feature', 'benefit', 'tags'])
    req['sentence'] = req['feature'] + ', ' + req['benefit']
    req.drop(['feature', 'benefit'], axis=1, inplace=True)
    req.dropna(inplace=True)
    req.reset_index(drop=True, inplace=True)
    req['tags'] = req['tags'].apply(lambda x: list(re.split(r'\\W+', x)))
    return req

def get_tag_count(req):
    return len(req['tags'])

if __name__ == "__main__":
    data = load_data('requirements.csv')
    print(f"Number of tags: {get_tag_count(data)}")
