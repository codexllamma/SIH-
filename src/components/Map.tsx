import React from "react";
import TrainSimulation from "./train-simulation";
import BigMap from "./BigMap"; // 👈 your 248 lines of map code

const Map = () => {
  return (
    <div className="bg-gray-800 p-4 rounded-xl shadow-lg border border-gray-700 w-full md:w-3/4">
      <h2 className="text-xl font-semibold mb-4 text-teal-300">
        Live Train Map
      </h2>
      <div className="bg-gray-700 rounded-lg h-100 flex">
        {/* Map area */}
        <div className="flex-1">
          <BigMap />
        </div>

        {/* Train Simulation area
        <div className="flex-1">
          <TrainSimulation />
        </div> */}
      </div>
    </div>
  );
};

export default Map;
