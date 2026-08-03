import sys
sys.path.insert(0, '/shared/mm-new')
from net_mutation.mutate_graph_demo import mutate_json_all
mutate_json_all(2, '/shared/mm-new/MindSpeed-MM/examples/internvl3/model_8B.json', '/shared/mm-new/mm_mutation_results/internvl3')
