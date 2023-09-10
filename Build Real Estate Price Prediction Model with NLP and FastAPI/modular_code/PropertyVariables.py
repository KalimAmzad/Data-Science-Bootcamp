from pydantic import BaseModel

class ProperyPricePred(BaseModel):
    PropertyType: float
    ClubHouse: float 
    School_University_in_Township: float 
    Hospital_in_TownShip: float  
    Mall_in_TownShip: float  
    Park_Jogging_track: float 
    Swimming_Pool: float 
    Gym: float  
    Property_Area_in_Sq_Ft: float  
    Price_by_sub_area: float 
    Amenities_score: float 
    Price_by_Amenities_score: float 
    Noun_Counts: float 
    Verb_Counts: float 
    Adjective_Counts: float 
    boasts_elegant: float 
    elegant_towers: float 
    every_day: float 
    great_community: float 
    mantra_gold: float 
    offering_bedroom: float 
    quality_specification: float 
    stories_offering: float 
    towers_stories: float 
    world_class: float 