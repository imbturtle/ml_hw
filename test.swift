import Foundation


//class is referencetype
class Item: NSObject {
  var name: String
  var weight: Double
  
  init(name: String, weight: Double) {
    self.name = name
    self.weight = weight
  }
}

class PersonalInformation: NSObject, NSCopying {
  
  var name: String
  var item: Item
  
  init(name: String, item:Item) {
    self.name = name
    self.item = item
  }
  
  func copy(with zone: NSZone? = nil) -> Any {
    return PersonalInformation(name: self.name, item: self.item)
  }
}

var eren = PersonalInformation(name: "Eren", item: Item(name: "Thunder Spear", weight: 50.0))
