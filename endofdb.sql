DROP DATABASE IF EXISTS endof;
CREATE DATABASE endof;
USE endof;

CREATE TABLE `ga_instances` (
  `instance_id` varchar(12) NOT NULL,
  `run_num` tinyint(4) NOT NULL,
  `is_multiverse` tinyint(4) NOT NULL,
  `num_nodes` smallint(6) NOT NULL,
  `num_cities` smallint(6) NOT NULL,
  `runtime_user` float NOT NULL,
  `runtime_system` float NOT NULL,
  `runtime_wall` float NOT NULL,
  `random_seed` float NOT NULL,
  `best_sol_end` float NOT NULL,
  `best_known_sol` float DEFAULT NULL,
  PRIMARY KEY (`instance_id`,`run_num`,`is_multiverse`)
);

CREATE TABLE `ga_iters` (
  `instance_id` varchar(12) NOT NULL,
  `run_num` tinyint(4) NOT NULL,
  `is_multiverse` tinyint(4) NOT NULL,
  `iter_num` smallint(6) NOT NULL,
  `best_sol` float NOT NULL,
  PRIMARY KEY (`instance_id`,`run_num`,`is_multiverse`,`iter_num`),
  KEY `ga_experiment` (`instance_id`,`run_num`,`is_multiverse`),
  CONSTRAINT `ga_experiment` FOREIGN KEY (`instance_id`, `run_num`, `is_multiverse`) REFERENCES `ga_instances` (`instance_id`, `run_num`, `is_multiverse`)
);

CREATE TABLE `aco_instances` (
  `instance_id` varchar(12) NOT NULL,
  `run_num` tinyint(4) NOT NULL,
  `is_multiverse` tinyint(4) NOT NULL,
  `num_nodes` smallint(6) NOT NULL,
  `num_cities` smallint(6) NOT NULL,
  `runtime_user` float NOT NULL,
  `runtime_system` float NOT NULL,
  `runtime_wall` float NOT NULL,
  `random_seed` float NOT NULL,
  `best_sol_end` float NOT NULL,
  `best_known_sol` float DEFAULT NULL,
  PRIMARY KEY (`instance_id`,`run_num`,`is_multiverse`)
);

CREATE TABLE `aco_iters` (
  `instance_id` varchar(12) NOT NULL,
  `run_num` tinyint(4) NOT NULL,
  `is_multiverse` tinyint(4) NOT NULL,
  `iter_num` smallint(6) NOT NULL,
  `best_sol` float NOT NULL,
  PRIMARY KEY (`instance_id`,`run_num`,`is_multiverse`,`iter_num`),
  KEY `aco_experiment` (`instance_id`,`run_num`,`is_multiverse`),
  CONSTRAINT `aco_experiment` FOREIGN KEY (`instance_id`, `run_num`, `is_multiverse`) REFERENCES `aco_instances` (`instance_id`, `run_num`, `is_multiverse`)
);

# Will fail here if user already exists (e.g. if resetting DB)
# If the user and its privileges already exist, the fail is inconsequential
CREATE USER 'endof'@'localhost' IDENTIFIED BY 'endof';
GRANT ALL ON endof.* to 'endof'@'localhost';
