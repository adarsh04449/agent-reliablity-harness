CREATE TABLE airports (
  code TEXT PRIMARY KEY,
  city TEXT NOT NULL
);

CREATE TABLE flights (
  id TEXT PRIMARY KEY,
  origin TEXT NOT NULL REFERENCES airports(code),
  destination TEXT NOT NULL REFERENCES airports(code),
  date TEXT NOT NULL,
  depart TEXT NOT NULL,
  time_of_day TEXT NOT NULL,
  price INTEGER NOT NULL,
  seats_left INTEGER NOT NULL,
  CHECK (origin != destination),
  CHECK (seats_left >= 0)
);

CREATE TABLE reservations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  flight_id TEXT NOT NULL REFERENCES flights(id),
  passenger_name TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'confirmed'
);

CREATE INDEX idx_flights_search ON flights (origin, destination, date);
CREATE INDEX idx_reservations_passenger ON reservations (passenger_name);
