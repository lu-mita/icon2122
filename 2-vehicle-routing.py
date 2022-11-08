import datetime

import decorator
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
from decimal import Decimal
import numpy as np
import pandas as pd
# https://graphhopper.com/


class Location:
    name: str = ""
    latitude: Decimal = 0
    longitude: Decimal = 0

    def __init__(self, latitude, longitude):
        self.latitude = latitude
        self.longitude = longitude

    def __str__(self):
        return "{name} @({latitude},{longitude})".format(
            name=self.name,
            latitude=self.latitude,
            longitude=self.longitude
        )


class Order:
    id: int = 0
    demand: int = 0
    # demand_volume: int = 0
    tw_open: datetime.timedelta
    tw_close: datetime.timedelta
    location: Location

    def __str__(self):
        return "Delivery[id={id},location={location}]".format(
            id=self.id,
            location=self.location
        )


class Vehicle:
    id: int = 0
    capacity: int = 0
    # capacity_volume: int = 0
    fixed_cost: int = 0
    start: int = 0
    end: int = 0

    def __str__(self):
        return "Vehicle[id={id},weight={weight},volume=]".format(
            id=self.id,
            weight=self.capacity  # , volume=self.capacity_volume
        )


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth specified in decimal degrees of latitude and longitude.
    https://en.wikipedia.org/wiki/Haversine_formula
    Args:
        lon1: longitude of pt 1,
        lat1: latitude of pt 1,
        lon2: longitude of pt 2,
        lat2: latitude of pt 2
    Returns:
        the distace in km between pt1 and pt2
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
    c = 2 * np.arcsin(np.sqrt(a))

    # 6367 km is the radius of the Earth
    km = 6367 * c
    return km


def compute_distance_matrix(deliveries: list[Order], distance_measure):
    matrix = np.zeros((len(deliveries), len(deliveries)))
    for i in deliveries:
        for j in deliveries:
            matrix[i.id][j.id] = distance_measure(
                i.location.latitude,
                i.location.longitude,
                j.location.latitude,
                j.location.longitude,
            )
    return matrix


class CVRPTWProblem:
    """
    Vehicle Routing Problem with Time Windows and Capacity constraints
    The optimization engine uses local search to improve solutions, first
    solutions being generated using a cheapest addition heuristic.
    """

    def __init__(self, orders, vehicles):
        self.assignement = None
        self.vehicles: list[Vehicle]
        self.orders: list[Order]
        # self.distance_matrix: np.ndarray
        self.manager: pywrapcp.RoutingIndexManager
        self.routing: pywrapcp.RoutingModel
        self.parameters: pywrapcp.DefaultRoutingSearchParameters

        self.vehicles = vehicles
        self.orders = orders
        self.distance_matrix = compute_distance_matrix(orders, haversine)
        # start = list({v.start for v in self.vehicles})
        # ends = list({v.end for v in self.vehicles})
        # self.manager = pywrapcp.RoutingIndexManager(len(self.deliveries), len(self.vehicles), start, ends)
        start = 0
        self.manager = pywrapcp.RoutingIndexManager(len(self.orders), len(self.vehicles), start)
        # Set model parameters
        model_parameters = pywrapcp.DefaultRoutingModelParameters()
        # Make the routing model instance.
        self.routing = pywrapcp.RoutingModel(self.manager, model_parameters)
        self.add_distance_constraint()
        self.add_time_constrain()
        self.add_capacity_constrain()

    def get_total_capacity(self):
        return sum((v.capacity for v in self.vehicles))

    def get_total_demand(self):
        return sum((d.demand for d in self.orders))

    def add_distance_constraint(self):
        """
        Set the cost function (distance callback) for each arc, homogeneous for
        all vehicles.
        """

        def distance(from_index, to_index):
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            return self.distance_matrix[from_node][to_node]

        distance_callback_index = self.routing.RegisterTransitCallback(distance)
        self.routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)
        # Set vehicle costs for each vehicle, not homogeneous.
        # for veh in vehicles.vehicles:
        #     routing.SetFixedCostOfVehicle(veh.cost, int(veh.index))

    def add_capacity_constrain(self):
        def demand(from_index):
            delivery_idx = self.manager.IndexToNode(from_index)
            return self.orders[delivery_idx].demand

        demand_callback_index = self.routing.RegisterUnaryTransitCallback(demand)

        # Add a dimension for vehicle capacities
        null_capacity_slack = 0
        self.routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            null_capacity_slack,
            [v.capacity for v in self.vehicles],  # capacity array
            True,
            'Capacity'
        )

    def add_time_constrain(self):
        def time(from_index, to_index):
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            service_time_per_dem = 300  # second
            # average speed in km/h
            speed_kmph = 10.0
            waiting_time = self.orders[from_node].demand * service_time_per_dem * 1.0
            travel_time = self.distance_matrix[from_node][to_node] / (speed_kmph * 1.0 / 60 ** 2)
            return waiting_time + travel_time

        time_callback_index = self.routing.RegisterTransitCallback(time)
        max_time_for_vehicles = max_waiting_time = 24 * 60 ** 2  # a day
        constrain_name = 'Time'
        self.routing.AddDimension(
            time_callback_index,
            max_waiting_time,
            max_time_for_vehicles,  # max time for vehicle
            False,  # Don't for force start cumulative time
            constrain_name
        )
        time_dimension = self.routing.GetDimensionOrDie(constrain_name)

        for delivery in self.orders:
            if delivery.location.name == 'depot':
                continue
            index = self.manager.NodeToIndex(delivery.id)
            time_dimension.CumulVar(index).SetRange(delivery.tw_open.seconds, delivery.tw_close.seconds)

    def solve(self):
        if self.get_total_demand() < self.get_total_demand():
            return
        parameters = pywrapcp.DefaultRoutingSearchParameters()

        # Setting first solution heuristic (cheapest addition).
        parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        # Routing: forbids use of TSPOpt neighborhood, (this is the default behaviour)
        parameters.local_search_operators.use_tsp_opt = pywrapcp.BOOL_FALSE
        # Disabling Large Neighborhood Search, (this is the default behaviour)
        parameters.local_search_operators.use_path_lns = pywrapcp.BOOL_FALSE
        parameters.local_search_operators.use_inactive_lns = pywrapcp.BOOL_FALSE

        parameters.time_limit.seconds = 10
        parameters.use_full_propagation = True
        # parameters.log_search = True

        # The solver parameters can be accessed from the model parameters. For example :
        #   model_parameters.solver_parameters.CopyFrom(
        #       pywrapcp.Solver.DefaultSolverParameters())
        #    model_parameters.solver_parameters.trace_propagation = True

        self.assignement = self.routing.SolveWithParameters(parameters)
        return self.assignement

    def vehicle_output_string(self):
        """
        Return a string displaying the output of the routing instance and
        assignment (plan).
        Args: routing (ortools.constraint_solver.pywrapcp.RoutingModel): routing.
        plan (ortools.constraint_solver.pywrapcp.Assignment): the assignment.
        Returns:
            (string) plan_output: describing each vehicle's plan.
            (List) dropped: list of dropped orders.
        """
        """
        Return a string displaying the output of the routing instance and
        assignment (plan).
        Args: routing (ortools.constraint_solver.pywrapcp.RoutingModel): routing.
        plan (ortools.constraint_solver.pywrapcp.Assignment): the assignment.
        Returns:
            (string) plan_output: describing each vehicle's self.assignement.
            (List) dropped: list of dropped orders.
        """
        dropped = []
        for order in range(self.routing.Size()):
            if self.assignement.Value(self.routing.NextVar(order)) == order:
                dropped.append(str(order))

        capacity_dimension = self.routing.GetDimensionOrDie('Capacity')
        time_dimension = self.routing.GetDimensionOrDie('Time')
        plan_output = ''

        for route_number in range(self.routing.vehicles()):
            order = self.routing.Start(route_number)
            plan_output += 'Route {0}:'.format(route_number)
            if self.routing.IsEnd(self.assignement.Value(self.routing.NextVar(order))):
                plan_output += ' Empty \n'
            else:
                while True:
                    load_var = capacity_dimension.CumulVar(order)
                    time_var = time_dimension.CumulVar(order)
                    node = self.manager.IndexToNode(order)
                    plan_output += \
                        ' {node} Load({load}) Time({tmin}, {tmax}) -> '.format(
                            node=node,
                            load=self.assignement.Value(load_var),
                            tmin=str(datetime.timedelta(seconds=self.assignement.Min(time_var))),
                            tmax=str(datetime.timedelta(seconds=self.assignement.Max(time_var)))
                        )

                    if self.routing.IsEnd(order):
                        plan_output += ' EndRoute {0}. \n'.format(route_number)
                        break
                    order = self.assignement.Value(self.routing.NextVar(order))
            plan_output += '\n'

        return plan_output, dropped


def import_data_model():
    df = pd.read_csv('data/deliveries.csv')
    deliveries = []
    for index, row in df.iterrows():
        d = Order()
        d.id = int(row['index'])
        d.location = Location(row['latitude'], row['longitude'])
        d.location.name = row['address']
        d.tw_open = datetime.timedelta(seconds=row['tw_open'])
        d.tw_close = datetime.timedelta(seconds=row['tw_close'])
        # d.demand_volume = row['demand_capacity']
        d.demand = int(row['demand_weight'])
        deliveries.append(d)
        print(d)

    df = pd.read_csv('data/vehicles.csv')
    vehicles = []
    for index, row in df.iterrows():
        v = Vehicle()
        v.id = int(row['index'])
        # v.capacity_volume = row['capacity_volume']
        v.capacity = int(row['capacity_weight'])
        v.start = 0
        v.end = 0
        # opt.ly implement a feature to indicate a different end point
        print(v)
        vehicles.append(v)

    return deliveries, vehicles


def main():
    orders, vehicles = import_data_model()
    csp = CVRPTWProblem(orders, vehicles)
    assignment = csp.solve()
    if assignment:
        r, p = csp.vehicle_output_string()
        print(r)
        print(p)


if __name__ == "__main__":
    main()

