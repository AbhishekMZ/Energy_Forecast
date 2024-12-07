@echo off

REM Move configuration files
move config.py energy_forecast\config\
move requirements.txt energy_forecast\
move setup.py energy_forecast\
move README.md energy_forecast\
move alembic.ini energy_forecast\config\
move Dockerfile energy_forecast\
move docker-compose.yml energy_forecast\

REM Move data files
move energy_data.csv energy_forecast\core\data\
move weather_features.csv energy_forecast\core\data\

REM Move notebooks
mkdir energy_forecast\notebooks
move Internship_AIML.ipynb energy_forecast\notebooks\
move polyReg.ipynb energy_forecast\notebooks\

REM Move Python scripts
move generate_training_data.py energy_forecast\scripts\
move run_pipeline.py energy_forecast\scripts\

REM Move directories
xcopy /E /I /Y alembic energy_forecast\alembic\
xcopy /E /I /Y api energy_forecast\api\
xcopy /E /I /Y app energy_forecast\app\
xcopy /E /I /Y auth energy_forecast\auth\
xcopy /E /I /Y config energy_forecast\config\
xcopy /E /I /Y database energy_forecast\database\
xcopy /E /I /Y frontend energy_forecast\frontend\
xcopy /E /I /Y models energy_forecast\core\models\
xcopy /E /I /Y scripts energy_forecast\scripts\
xcopy /E /I /Y src energy_forecast\core\src\
xcopy /E /I /Y tests energy_forecast\tests\
xcopy /E /I /Y utils energy_forecast\core\utils\

REM Clean up old directories after copying
rd /s /q alembic
rd /s /q api
rd /s /q app
rd /s /q auth
rd /s /q config
rd /s /q database
rd /s /q frontend
rd /s /q models
rd /s /q scripts
rd /s /q src
rd /s /q tests
rd /s /q utils

REM Remove reorganization scripts
del reorganize.py
del reorganize.bat

echo Files have been moved to the energy_forecast directory structure!
