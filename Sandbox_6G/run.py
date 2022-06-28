#import solar_ai_trust_v1
import nextg_ai_trust_v1


def run_experiment_1(params):
    print('======== PARAMS start =========')
    print(params)
    result = nextg_ai_trust_v1.run_all(params)
    print('======== PARAMS end =========')
    return result