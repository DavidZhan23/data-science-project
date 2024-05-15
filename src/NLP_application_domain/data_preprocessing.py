import pandas as pd
import re
from nltk.corpus import stopwords

def load_data(file_path):
    requirements = pd.read_csv(file_path)
    req = pd.DataFrame(requirements, columns=['feature', 'benefit', 'application_domain', 'application_domain_other'])
    req['sentence'] = req['feature'] + ', ' + req['benefit']
    req.drop(['feature', 'benefit'], axis=1, inplace=True)
    return req

def preprocess_application_domain(req, keep_list):
    for other, index in zip(req['application_domain_other'], range(len(req['application_domain_other']))):
        if type(other) != float:
            req['application_domain_other'][index] = other.lower()
    
    for domain, index in zip(req['application_domain'], range(len(req['application_domain']))):
        if domain != 'Other':
            new_domain = domain.lower()
            req['application_domain'][index] = new_domain
        elif domain == 'Other':
            if req['application_domain_other'][index] in keep_list:
                req['application_domain'][index] = req['application_domain_other'][index]
            else:
                req.drop(index, inplace=True)
    
    req.reset_index(drop=True, inplace=True)
    req.drop(['application_domain_other'], axis=1, inplace=True)
    return req
