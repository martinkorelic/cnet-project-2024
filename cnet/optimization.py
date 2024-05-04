import json
import numpy as np
from mealpy.swarm_based import PSO
from mealpy import Problem, FloatVar
from numpy import ndarray
from .data.filter import CNetRelations, CNetFilter
from .data.db import CNetDatabase
from .graph import CNetGraph
from .metrics import iou, positionaliou, advancediou, accuracy

class CNetAlgoProblem(Problem):
    """
    Mealpy Multi-objective Problem class.
    Optimize with respect to normal IoU, positional IoU and advanced IoU scores.
    """

    def __init__(self, name, query, db, solution_models, cnet_relations: CNetRelations, bounds, graph_path='graphdata', solution_path='wordsdata', minmax: str = "max", **kwargs) -> None:
        self.name = name
        self.query = query
        self.cnet_relations = cnet_relations
        self.solution_models = solution_models
        self.cnet_graph = CNetGraph(db, CNetFilter(cnet_relations))
        self.local_graph = self.cnet_graph.load_from_file(f'{graph_path}/{query}.graphml')

        # Collect solutions from wordsdata based on query
        with open(f'{solution_path}/{query}_words.json', 'r', encoding='utf-8') as file:
            self.data = json.load(file)

        super().__init__(bounds, minmax, **kwargs)

    def positionalIoU(self, x, y):
        return positionaliou(x, y)
    
    def IoU(self, x, y):
        return iou(x, y)
    
    def advancedIoU(self, x, y):
        return advancediou(x, y)
    
    def obj_func(self, x: ndarray):
        # Algorithm edge weights
        x = x / np.sum(x)

        self.cnet_relations.weights = { k : x[i] for i, (k, _) in enumerate(self.cnet_relations.weights.items()) }

        # TODO
        # Run algorithm with edge weights given
        print(self.cnet_relations.weights)
        
        # Algorithm random walk clustering
        if self.name == 'rwc':
            predicted = self.cnet_graph.random_walk_clustering(self.local_graph, root=self.query, etf=self.cnet_relations.weights, top_k=100)

        # Average the solution metrics
        iou_scores = np.zeros(len(self.solution_models))
        posiou_scores = np.zeros(len(self.solution_models))
        adviou_scores = np.zeros(len(self.solution_models))

        for i, sol_mod in enumerate(self.solution_models):
            iou_scores[i] = self.IoU(self.data[sol_mod], predicted)
            posiou_scores[i] = self.positionalIoU(self.data[sol_mod], predicted)
            adviou_scores[i] = self.advancedIoU(self.data[sol_mod], predicted)
            print(f'Algo: {sol_mod} Scores - IoU: {iou_scores[i]} | PosIoU: {posiou_scores[i]} | AdvIoU: {adviou_scores[i]} | Acc: {accuracy(self.data[sol_mod], predicted)}')

        return [np.average(iou_scores), np.average(posiou_scores), np.average(adviou_scores)]

def optimize_cnet_algo(query, algo_name, solution_models, db : CNetDatabase, cnet_relations : CNetRelations, epochs=20, n_workers=1):

    cnet_problem = CNetAlgoProblem(
        # Bounds should be equal to length of different edge weights
        name=algo_name,
        query=query,
        db=db,
        # Which model solutions to consider
        solution_models=solution_models,
        cnet_relations=cnet_relations,
        bounds=FloatVar(lb=[0 for _ in range(len(cnet_relations.weights))], ub=[1 for _ in range(len(cnet_relations.weights))]),
        obj_weights=[0.5, 0.25, 0.25]
    )

    problem_model = PSO.OriginalPSO(epoch=epochs)
    opt = problem_model.solve(problem=cnet_problem, n_workers=n_workers)
    
    # TODO: Remap best solution back to CNetRelations

    return opt.solution