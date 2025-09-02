import React from "react";

type AlertCardProps = {
  title: string;
  message: string;
  trainId: string;
  onAccept?: () => void;
  onDismiss?: () => void;
};

const AlertCard: React.FC<AlertCardProps> = ({
  title,
  message,
  trainId,
  onAccept,
  onDismiss,
}) => {

  return (
    <div className="bg-gray-700 p-4 rounded-lg shadow-md">
      <div className="flex justify-between items-start mb-2">
        <span className="text-md font-bold text-yellow-300 uppercase">
          <p>Train {trainId}</p>
        </span>
      </div>

      <p className="text-gray-300 text-sm mb-2 font-semibold">{title}</p>
      <p className="text-sm text-gray-100">{message}</p>

      <div className="mt-3 flex space-x-2">
        <button
          onClick={onAccept}
          className="flex-1 bg-green-600 hover:bg-green-700 text-white text-sm font-semibold py-2 rounded-lg transition-colors duration-200"
        >
          Accept
        </button>
        <button
          onClick={onDismiss}
          className="flex-1 bg-red-600 hover:bg-red-700 text-white text-sm font-semibold py-2 rounded-lg transition-colors duration-200"
        >
          Dismiss
        </button>
      </div>
    </div>
  );
};

export default AlertCard;