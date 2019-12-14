-- --------------------------------------------------------
-- Host:                         192.168.1.21
-- Server version:               8.0.17 - MySQL Community Server - GPL
-- Server OS:                    Linux
-- HeidiSQL Version:             10.1.0.5464
-- --------------------------------------------------------

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET NAMES utf8 */;
/*!50503 SET NAMES utf8mb4 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;


-- Dumping database structure for knowage-db
CREATE DATABASE IF NOT EXISTS `knowage-db` /*!40100 DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci */ /*!80016 DEFAULT ENCRYPTION='N' */;
USE `knowage-db`;

-- Dumping structure for table knowage-db.weather_data
CREATE TABLE IF NOT EXISTS `weather_data` (
  `2t` float DEFAULT NULL,
  `sf` float DEFAULT NULL,
  `sp` float DEFAULT NULL,
  `tcc` float DEFAULT NULL,
  `tp` float DEFAULT NULL,
  `ws` float DEFAULT NULL,
  `rh` float DEFAULT NULL,
  `region` varchar(50) DEFAULT NULL,
  `strRegion` int(11) DEFAULT NULL,
  `geonameId` int(11) DEFAULT NULL,
  `validTime` datetime DEFAULT NULL,
  `validityDateTime` datetime DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Data exporting was unselected.
/*!40101 SET SQL_MODE=IFNULL(@OLD_SQL_MODE, '') */;
/*!40014 SET FOREIGN_KEY_CHECKS=IF(@OLD_FOREIGN_KEY_CHECKS IS NULL, 1, @OLD_FOREIGN_KEY_CHECKS) */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
