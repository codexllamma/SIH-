import React from "react";

interface HeaderProps {
  section: string;
  controller: string;
  status: "Operational" | "Down" | "Maintenance"; 
  onRefresh?: () => void; 
}

const Header: React.FC<HeaderProps> = ({ section, controller, status, onRefresh }) => {
  return (
    <header className="bg-gray-800 p-4 rounded-xl shadow-lg border border-gray-700">
      <div className="flex justify-between items-center flex-wrap">
        <div>
          <h1 className="text-2xl font-bold text-teal-400">
            Section Throughput Optimizer
          </h1>
          <p className="text-sm text-gray-400 mt-1">
            Section: {section} | Controller: {controller}
          </p>
        </div>
        <div className="flex items-center space-x-4 mt-2 sm:mt-0">
          <span className="text-sm text-gray-400">
            Status:{" "}
            <span className="text-green-400 font-medium">{status}</span>
          </span>
          <button
            onClick={onRefresh ? onRefresh : undefined}
            className="bg-red-600 hover:bg-red-700 text-white font-semibold py-2 px-4 rounded-lg shadow-md transition-colors duration-200"
          >
            Refresh
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header;
