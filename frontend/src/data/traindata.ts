export const tracksData = [
  { id: "T1", points: [0, 80, 1000, 80] }, // line 1
  { id: "T2", points: [0, 200, 1000, 200] }, // line 2
  { id: "T3", points: [0, 320, 1000, 320] }, // line 3
  { id: "T4", points: [300, 80, 500, 200] }, // branch track
];

export const junctions = [
  { id: "J1", x: 300, y: 80, connectedTracks: ["T1", "T2", "T4"] }, // include T4
  { id: "J2", x: 500, y: 200, connectedTracks: ["T2", "T4"] }, // include T4
];

export const trains = 
[
  {
    "id": "T101",
    "label": "Rajdhani Express",
    "colour": "red",
    "x": 0,
    "y": 50,
    "speed": 2.5,
    "delay": 0,
    "route": ["Delhi", "Kanpur", "Prayagraj", "Patna", "Howrah"]
  },
  {
    "id": "T202",
    "label": "Shatabdi Express",
    "colour": "blue",
    "x": 0,
    "y": 150,
    "speed": 3.0,
    "delay": 10,
  },
  {
    "id": "T303",
    "label": "Duronto Express",
    "colour": "green",
    "x": 0,
    "y": 250,
    "speed": 2.0,
    "delay": 5,
    
  },
  {
    "id": "T404",
    "label": "Garib Rath",
    "colour": "orange",
    "x": 0,
    "y": 350,
    "speed": 1.8,
    "delay": 0,
    
  }
]

export const suggestions = 
[
  {
    "trainId": "T2",
    "suggestionId": "S1",
    "type": "reroute",
    "message": "Reroute T2 from track 1 to 2",
    "impact": -8
  },
  {
    "trainId": "T3",
    "suggestionId": "S2",
    "type": "hold",
    "message": "Hold T3 at P1 to let T1 pass.",
    "impact": 5
  },
  {
    "trainId": "T1",
    "suggestionId": "S3",
    "type": "priority",
    "message": "Give T1 priority in crossing junction.",
    "impact": -12
  }
]

export const delays = 
[
  { "trainId": "T101", "station": "Kanpur", "scheduled": "12:30", "actual": "12:45", "delay": 15 },
  { "trainId": "T202", "station": "Jhansi", "scheduled": "14:10", "actual": "14:25", "delay": 15 },
  { "trainId": "T303", "station": "Vadodara", "scheduled": "16:00", "actual": "16:05", "delay": 5 }
]

export const platforms = [
  { id: "APF1", label: "A/PF1", colour: "lightblue", x: 50, y: 110 },
  { id: "APF2", label: "A/PF2", colour: "lightblue", x: 50, y: 240 },
  { id: "BPF1", label: "B/PF1", colour: "lavender", x: 720, y: 110 },
  { id: "BPF2", label: "B/PF2", colour: "lavender", x: 720, y: 240 },
];

// signals
export const signals = [
  { id: 1, x: 300, y: 80 },
  { id: 2, x: 500, y: 200 },
  { id: 3, x: 150, y: 80 },
  { id: 4, x: 150, y: 200 },
  { id: 5, x: 820, y: 80 },
  { id: 6, x: 820, y: 200 },
];

// 👇 NEW: train routes (sequence of track IDs each train follows)
export const trainRoutes: { [trainId: string]: string[] } = {
  T1: ["T1", "T4", "T2"], // spawn on T1, branch to T4, then continue on T2
  T2: ["T2"], // stays on T2
  T3: ["T3"], // stays on T3
};

/*
const pastelColors = {
  pink: "#FADADD",      // 
  peach: "#FFDAB9",     // pastel peach
  mint: "#B9FBC0",      // mint green
  babyBlue: "#A7C7E7",  // baby blue
  lavender: "#E6E6FA",  // lavender
  lemon: "#FFFACD",     // pastel yellow
  coral: "#F6D7B0",     // soft coral
  lilac: "#D8BFD8",     // lilac purple
};
*/
