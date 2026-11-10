-- Serving store for the recommendation lab. Everything lives in its own schema
-- (default "recsys"; the loader substitutes the name), so other tables in the database
-- are untouched. Impressions (~2M rows) stay in parquet: they are only used offline.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE SCHEMA IF NOT EXISTS {schema};
SET search_path TO {schema}, public;

CREATE TABLE cities (
    code text PRIMARY KEY, name text NOT NULL, lat double precision, lon double precision
);
CREATE TABLE districts (
    code text PRIMARY KEY, city text NOT NULL REFERENCES cities(code), name text NOT NULL,
    lat double precision, lon double precision, density real, income real
);
CREATE TABLE restaurants (
    restaurant_id int PRIMARY KEY,
    city text NOT NULL REFERENCES cities(code),
    district text NOT NULL REFERENCES districts(code),
    name text NOT NULL, tagline text, cuisine text NOT NULL, chain text,
    price_tier smallint NOT NULL CHECK (price_tier BETWEEN 1 AND 3),
    rating real, n_ratings int,
    lat double precision NOT NULL, lon double precision NOT NULL,
    geog geography(Point, 4326),
    delivery_radius_km double precision NOT NULL, prep_min smallint, hours jsonb NOT NULL
);
CREATE TABLE items (
    item_id int PRIMARY KEY,
    restaurant_id int NOT NULL REFERENCES restaurants(restaurant_id),
    sku text NOT NULL, name text NOT NULL, description text,
    category text NOT NULL CHECK (category IN ('main', 'side', 'drink', 'dessert')),
    price numeric(8, 2) NOT NULL CHECK (price > 0),
    vegetarian boolean NOT NULL, vegan boolean NOT NULL, spicy boolean NOT NULL, tags text[]
);
CREATE TABLE users (
    user_id int PRIMARY KEY, home_city text NOT NULL REFERENCES cities(code),
    district text NOT NULL REFERENCES districts(code),
    home_lat double precision, home_lon double precision,
    work_lat double precision, work_lon double precision,
    signup_date date NOT NULL, platform text, channel text, age_band text
);
CREATE TABLE weather (
    date date NOT NULL, city text NOT NULL REFERENCES cities(code),
    rain boolean NOT NULL, temp_max_c real, PRIMARY KEY (city, date)
);
CREATE TABLE sessions (
    session_id bigint PRIMARY KEY, user_id int NOT NULL REFERENCES users(user_id),
    ts timestamp NOT NULL, lat double precision, lon double precision, city text,
    location_kind text, daypart smallint, rain boolean, hot boolean,
    n_available int, ordered boolean
);
CREATE TABLE orders (
    order_id bigint PRIMARY KEY, session_id bigint NOT NULL REFERENCES sessions(session_id),
    user_id int NOT NULL REFERENCES users(user_id), ts timestamp NOT NULL,
    restaurant_id int NOT NULL REFERENCES restaurants(restaurant_id), city text,
    lat double precision, lon double precision,
    distance_km real, delivery_fee numeric(6, 2), subtotal numeric(8, 2)
);
CREATE TABLE order_items (
    order_id bigint NOT NULL REFERENCES orders(order_id),
    item_id int NOT NULL REFERENCES items(item_id),
    qty smallint NOT NULL, unit_price numeric(8, 2) NOT NULL, is_main boolean NOT NULL
);
