import React from "react";

const InfoBoxes = () => {
  return (
    <div className="bg-gray-900 p-6 rounded-xl shadow-lg">
      <h2 className="text-xl font-semi-bolc text-teal-300 mb-4">
        System Overview
      </h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-6">
        {/* Active Trains */}
        <div className="bg-gray-700 p-6 rounded-xl text-center shadow-md">
          <h3 className="text-teal-400 font-semibold">Active Trains</h3>
          <p className="text-2xl font-bold text-white">12</p>
        </div>

        {/* Delayed Trains */}
        <div className="bg-gray-700 p-6 rounded-xl text-center shadow-md">
          <h3 className="text-yellow-400 font-semibold">Delayed Trains</h3>
          <p className="text-2xl font-bold text-white">3</p>
        </div>

        {/* Signals Active */}
        <div className="bg-gray-700 p-6 rounded-xl text-center shadow-md">
          <h3 className="text-red-400 font-semibold">Signals Active</h3>
          <p className="text-2xl font-bold text-white">5</p>
        </div>

        {/* Upcoming Departures */}
        <div className="bg-gray-700 p-6 rounded-xl text-center shadow-md">
          <h3 className="text-purple-400 font-semibold">Upcoming Departures</h3>
          <p className="text-2xl font-bold text-white">8</p>
        </div>
      </div>
    </div>
  );
};

export default InfoBoxes;
